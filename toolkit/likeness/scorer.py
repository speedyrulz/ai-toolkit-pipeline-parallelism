"""Trainer-side launcher for the character likeness scoring worker.

The worker (score_worker.py) must run inside a ComfyUI installation's Python,
because it reuses ComfyUI's SAM3D Body implementation and the character
similarity custom nodes. This class starts it once per training run, hands it
one job per sample round, and collects results without blocking training.

ComfyUI may live in WSL while training runs on Windows: a comfyui_path like
\\\\wsl.localhost\\<distro>\\home\\me\\ComfyUI is launched through wsl.exe,
and every Windows path handed to the worker is translated to /mnt/<drive>/...
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import List, Optional

from toolkit.print import print_acc

RESULT_PREFIX = "LIKENESS_RESULT "
READY_PREFIX = "LIKENESS_READY"

_WSL_UNC = re.compile(r"^[\\/]{2}wsl(?:\.localhost|\$)[\\/]+([^\\/]+)[\\/]*(.*)$", re.IGNORECASE)


def _to_wsl_path(path: str) -> str:
    """Windows path -> path inside WSL."""
    path = os.path.abspath(path)
    m = _WSL_UNC.match(path)
    if m:
        return "/" + m.group(2).replace("\\", "/")
    drive, rest = os.path.splitdrive(path)
    if drive and drive[1:] == ":":
        return f"/mnt/{drive[0].lower()}" + rest.replace("\\", "/")
    return path.replace("\\", "/")


class LikenessScorer:
    def __init__(self, config, out_dir: str):
        self.config = config
        self.out_dir = out_dir
        self.proc: Optional[subprocess.Popen] = None
        self.results: "queue.Queue[dict]" = queue.Queue()
        self.pending: List[int] = []
        self.ready = False
        self.failed: Optional[str] = None
        self._stderr_file = None

    # ------------------------------------------------------------------
    def _command(self) -> List[str]:
        c = self.config
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_worker.py")
        m = _WSL_UNC.match(c.comfyui_path) if sys.platform == "win32" else None
        use_wsl = m is not None
        if use_wsl:
            distro = m.group(1)
            comfy_root = "/" + m.group(2).replace("\\", "/").rstrip("/")
            conv = _to_wsl_path
        else:
            comfy_root = os.path.abspath(c.comfyui_path)
            conv = os.path.abspath

        python = c.python
        if not python:
            if use_wsl or sys.platform != "win32":
                python = f"{comfy_root}/venv/bin/python"
            else:
                python = os.path.join(comfy_root, "venv", "Scripts", "python.exe")

        args = [
            python, "-u", conv(worker) if use_wsl else worker,
            "--comfyui", comfy_root,
            "--refs", conv(c.reference_folder),
            "--out", conv(self.out_dir),
            "--device", c.device,
            "--cpu-threads", str(c.cpu_threads),
            "--sam3d-model", c.sam3d_model,
            "--clip-vision-model", c.clip_vision_model,
            "--face-library", c.face_library,
            "--proportion-tolerance", str(c.proportion_tolerance),
            "--build-tolerance", str(c.build_tolerance),
            "--proportion-weight", str(c.proportion_weight),
            "--weights", ",".join(str(w) for w in c.weights),
        ]
        if use_wsl:
            return ["wsl.exe", "-d", distro, "--cd", comfy_root, "--"] + args
        return args

    def _to_worker_path(self, path: str) -> str:
        if sys.platform == "win32" and _WSL_UNC.match(self.config.comfyui_path):
            return _to_wsl_path(path)
        return os.path.abspath(path)

    # ------------------------------------------------------------------
    def start(self):
        os.makedirs(self.out_dir, exist_ok=True)
        cmd = self._command()
        print_acc(f"Starting likeness scorer ({self.config.device}); log: "
                  f"{os.path.join(self.out_dir, 'likeness_worker.log')}")
        self._stderr_file = open(os.path.join(self.out_dir, "likeness_worker.log"), "a",
                                 encoding="utf-8", errors="replace")
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith(READY_PREFIX):
                self.ready = True
            elif line.startswith(RESULT_PREFIX):
                try:
                    result = json.loads(line[len(RESULT_PREFIX):])
                except json.JSONDecodeError:
                    continue
                if "fatal" in result:
                    self.failed = result["fatal"]
                self.results.put(result)
        if not self.failed and self.pending:
            self.failed = f"worker exited (code {self.proc.poll()})"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.failed is None

    def submit(self, step: int, folder: str):
        if not self.alive():
            return False
        job = {"step": int(step), "folder": self._to_worker_path(folder)}
        try:
            self.proc.stdin.write(json.dumps(job) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self.failed = f"could not reach worker: {e}"
            return False
        self.pending.append(int(step))
        return True

    def poll(self) -> List[dict]:
        out = []
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            step = result.get("step")
            if step in self.pending:
                self.pending.remove(step)
            out.append(result)
        return out

    def finish(self, timeout_s: float) -> List[dict]:
        """Wait (up to timeout_s) for queued rounds, then stop the worker."""
        out = self.poll()
        deadline = time.time() + max(0.0, timeout_s)
        if self.pending and self.alive():
            print_acc(f"Waiting up to {timeout_s / 60:.0f} min for likeness scoring of "
                      f"step(s) {', '.join(str(s) for s in self.pending)}")
        while self.pending and self.alive() and time.time() < deadline:
            time.sleep(2)
            out += self.poll()
        self.close()
        out += self.poll()
        return out

    def close(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.stdin.close()
                try:
                    self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except Exception:
            pass
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
