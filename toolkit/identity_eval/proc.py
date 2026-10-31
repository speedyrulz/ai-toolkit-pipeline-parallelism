"""JSON-lines request/response over a child process's stdin/stdout.

Used for the identity eval scoring workers and the sample helper. Protocol
lines on the child's stdout carry PREFIX; anything else a library prints is
copied to the log file. Requests carry an integer id that the reply echoes, so
a caller can send to several children and then wait on each in turn.

Child side helpers (serve/reply) live here too and import nothing beyond the
standard library, so the evaluator environment can use them by path.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

PREFIX = "IDEVAL "


class ChildError(RuntimeError):
    pass


class JsonProcess:
    def __init__(self, name: str, cmd: List[str], log_path: str, env: Optional[Dict[str, str]] = None,
                 cwd: Optional[str] = None):
        self.name = name
        self.cmd = cmd
        self.log_path = log_path
        self.env = env
        self.cwd = cwd
        self.proc: Optional[subprocess.Popen] = None
        self._log = None
        self._replies: Dict[int, dict] = {}
        self._cond = threading.Condition()
        self._next_id = 1
        self._ready = False
        self._exit_reason: Optional[str] = None

    # ------------------------------------------------------------------
    def start(self, ready_timeout: float = 1800.0):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} starting {self.name}: {' '.join(self.cmd)}\n")
        self._log.flush()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=self.env,
                cwd=self.cwd,
                creationflags=creationflags,
            )
        except OSError as e:
            # a missing interpreter (evaluator_python typo) is a ChildError like
            # any other start failure, so callers can disable the feature
            self._log.close()
            self._log = None
            raise ChildError(f"{self.name} could not start ({e}): {self.cmd[0]!r}") from e
        threading.Thread(target=self._read_loop, daemon=True).start()
        deadline = time.time() + ready_timeout
        with self._cond:
            while not self._ready:
                if self._exit_reason is not None:
                    raise ChildError(f"{self.name} exited during startup: {self._exit_reason} (see {self.log_path})")
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise ChildError(f"{self.name} did not become ready in {ready_timeout:.0f}s (see {self.log_path})")
                self._cond.wait(min(remaining, 1.0))

    def _read_loop(self):
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if line.startswith(PREFIX):
                try:
                    msg = json.loads(line[len(PREFIX):])
                except json.JSONDecodeError:
                    self._write_log(line)
                    continue
                with self._cond:
                    if msg.get("event") == "ready":
                        self._ready = True
                    elif "id" in msg:
                        self._replies[int(msg["id"])] = msg
                    self._cond.notify_all()
            else:
                self._write_log(line)
        code = self.proc.wait()
        with self._cond:
            self._exit_reason = f"exit code {code}"
            self._cond.notify_all()

    def _write_log(self, line: str):
        try:
            self._log.write(line)
            self._log.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def send(self, cmd: str, **payload) -> int:
        if not self.alive:
            raise ChildError(f"{self.name} is not running ({self._exit_reason or 'not started'})")
        with self._cond:
            req_id = self._next_id
            self._next_id += 1
        line = json.dumps({"id": req_id, "cmd": cmd, **payload})
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            # the child died between the alive check and the write
            raise ChildError(f"{self.name} pipe closed: {e}") from e
        return req_id

    def wait(self, req_id: int, timeout: Optional[float] = None) -> dict:
        deadline = None if timeout is None else time.time() + timeout
        with self._cond:
            while req_id not in self._replies:
                if self._exit_reason is not None:
                    raise ChildError(f"{self.name} exited: {self._exit_reason} (see {self.log_path})")
                if deadline is not None and time.time() >= deadline:
                    raise ChildError(f"{self.name} did not answer within {timeout:.0f}s")
                self._cond.wait(1.0)
            msg = self._replies.pop(req_id)
        if not msg.get("ok", False):
            raise ChildError(f"{self.name}: {msg.get('error', 'unknown error')}")
        return msg

    def poll(self, req_id: int) -> Optional[dict]:
        """Reply if it has arrived (raises on a failed reply), else None."""
        with self._cond:
            if req_id not in self._replies:
                if self._exit_reason is not None:
                    raise ChildError(f"{self.name} exited: {self._exit_reason} (see {self.log_path})")
                return None
        return self.wait(req_id, timeout=0)

    def request(self, cmd: str, timeout: Optional[float] = None, **payload) -> dict:
        return self.wait(self.send(cmd, **payload), timeout)

    def close(self, timeout: float = 30.0):
        if self.proc is None:
            return
        if self.alive:
            try:
                self.send("shutdown")
                self.proc.wait(timeout)
            except Exception:
                pass
        if self.alive:
            self.proc.kill()
        if self._log is not None:
            try:
                self._log.close()
            except Exception:
                pass
        self.proc = None


# ----------------------------------------------------------------------
# child side
# ----------------------------------------------------------------------

def emit(msg: dict):
    sys.stdout.write(PREFIX + json.dumps(msg) + "\n")
    sys.stdout.flush()


def serve(handlers: Dict[str, Callable[..., dict]]):
    """Child main loop: announce ready, then answer one request per stdin
    line until shutdown or EOF. A handler returns a dict merged into the
    reply; an exception becomes ok=false with the message."""
    import traceback

    emit({"event": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        req_id = req.pop("id", None)
        cmd = req.pop("cmd", None)
        if cmd == "shutdown":
            emit({"id": req_id, "ok": True})
            return
        handler = handlers.get(cmd)
        if handler is None:
            emit({"id": req_id, "ok": False, "error": f"unknown command {cmd!r}"})
            continue
        try:
            out = handler(**req) or {}
            emit({"id": req_id, "ok": True, **out})
        except Exception as e:
            traceback.print_exc()
            emit({"id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}"})
