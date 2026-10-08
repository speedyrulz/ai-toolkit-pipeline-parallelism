"""Merged-model export of OmniVoice LoRAs for ComfyUI.

The ComfyUI-OmniVoice-TTS nodes cannot load LoRA files; they list every
folder in ``models/omnivoice`` that holds a model. An export is such a folder:
the base model's files with the LoRA folded into ``model.safetensors``, and the
769 MB audio tokenizer linked rather than copied.

Used by the trainer after each save (``comfyui_export_dir``) and, for LoRAs
that are already trained, by ``scripts/export_omnivoice_lora.py``. Has no
toolkit imports so the script stays light.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Callable, Optional

import torch

EXPORT_MARKER = "aitk_omnivoice_export.json"
# files of a model folder that are copied (small) or linked (large) on export
LINK_DIRS = ("audio_tokenizer",)
SKIP_FILES = ("model.safetensors", "README.md", ".gitattributes")
DTYPES = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16,
          "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}

_WSL_UNC = re.compile(r"^[\\/]{2}wsl(?:\.localhost|\$)[\\/]+([^\\/]+)[\\/]*(.*)$", re.IGNORECASE)


def wsl_parts(path: str):
    """(distro, linux_path) for a \\\\wsl.localhost path, else None."""
    m = _WSL_UNC.match(path)
    if not m:
        return None
    return m.group(1), "/" + m.group(2).replace("\\", "/").rstrip("/")


def to_linux_path(path: str, distro: str) -> Optional[str]:
    parts = wsl_parts(path)
    if parts is not None:
        return parts[1] if parts[0].lower() == distro.lower() else None
    drive, rest = os.path.splitdrive(os.path.abspath(path))
    if drive and drive[1:] == ":":
        return f"/mnt/{drive[0].lower()}" + rest.replace("\\", "/")
    return None


@torch.no_grad()
def merge_lora_file(base_path: str, lora_path: str, dtype: torch.dtype = torch.float32) -> dict:
    """Base model.safetensors with a saved LoRA file folded in. Keys are
    relative to the OmniVoice root (llm.layers.N.self_attn.q_proj...). PEFT
    format (lora_A/lora_B) carries no alpha, so its scale is 1; kohya format
    (lora_down/lora_up/.alpha) uses alpha / rank."""
    from safetensors.torch import load_file

    base = load_file(os.path.join(base_path, "model.safetensors"))
    lora = load_file(lora_path)
    modules = set()
    for k in lora:
        m = re.match(r"^(.*)\.(lora_A|lora_B|lora_down|lora_up)\.weight$", k)
        if m:
            modules.add(m.group(1))
        elif not k.endswith(".alpha"):
            raise ValueError(f"{os.path.basename(lora_path)}: unsupported key {k} (only plain LoRA can be merged)")
    if not modules:
        raise ValueError(f"{os.path.basename(lora_path)}: no LoRA weights found")
    for mod in sorted(modules):
        if f"{mod}.lora_A.weight" in lora:
            down, up = lora[f"{mod}.lora_A.weight"], lora[f"{mod}.lora_B.weight"]
        else:
            down, up = lora[f"{mod}.lora_down.weight"], lora[f"{mod}.lora_up.weight"]
        scale = 1.0
        if f"{mod}.alpha" in lora:
            scale = float(lora[f"{mod}.alpha"]) / down.shape[0]
        key = f"{mod}.weight"
        if key not in base:
            raise ValueError(f"{os.path.basename(lora_path)}: {key} is not in the base model; wrong base?")
        base[key] = base[key].float() + scale * (up.float() @ down.float())
    return {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in base.items()}


def write_export(state: dict, lora_path: str, base_path: str, export_dir: str, keep: int = 0,
                 log: Callable[[str], None] = print) -> str:
    """Write ``state`` as a model folder named after the LoRA file into
    ``export_dir``. Returns the folder."""
    from safetensors.torch import save_file

    t0 = time.time()
    name = os.path.splitext(os.path.basename(lora_path))[0]
    dest = os.path.join(export_dir, name)
    os.makedirs(dest, exist_ok=True)
    dest_wsl = wsl_parts(dest) if sys.platform == "win32" else None
    copied = False
    if dest_wsl is not None:
        # Windows writes into a \\wsl.localhost share run at ~6 MB/s; a
        # copy run inside WSL from /mnt/c measured ~150 MB/s (25x). Stage
        # the weights on local disk next to the LoRA, then copy from WSL.
        stage = os.path.join(os.path.dirname(os.path.abspath(lora_path)), "_export_staging", name)
        os.makedirs(stage, exist_ok=True)
        staged = os.path.join(stage, "model.safetensors")
        save_file(state, staged, metadata={"format": "pt"})
        del state
        distro, dest_linux = dest_wsl
        src_linux = to_linux_path(staged, distro)
        if src_linux:
            r = subprocess.run(["wsl.exe", "-d", distro, "--", "cp", src_linux, dest_linux + "/model.safetensors"],
                               capture_output=True)
            copied = r.returncode == 0
        if not copied:
            shutil.copyfile(staged, os.path.join(dest, "model.safetensors"))
        shutil.rmtree(stage, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(stage))  # only if no other export is staging
        except OSError:
            pass
    else:
        save_file(state, os.path.join(dest, "model.safetensors"), metadata={"format": "pt"})
        del state
    written = ["model.safetensors"]
    for entry in sorted(os.listdir(base_path)):
        src = os.path.join(base_path, entry)
        if os.path.isfile(src) and entry not in SKIP_FILES and not entry.startswith("."):
            shutil.copy2(src, os.path.join(dest, entry))
            written.append(entry)
    linked = {}
    for d in LINK_DIRS:
        src = os.path.join(base_path, d)
        if os.path.isdir(src):
            linked[d] = link_dir(src, os.path.join(dest, d))
    with open(os.path.join(dest, EXPORT_MARKER), "w", encoding="utf-8") as f:
        json.dump({"source_lora": os.path.abspath(lora_path), "files": written, "dirs": linked}, f, indent=2)
    log(f"omnivoice: exported merged model to {dest} ({time.time() - t0:.0f}s)")
    if keep > 0:
        prune_exports(export_dir, name, keep, log)
    return dest


def link_dir(src: str, dst: str) -> str:
    """Link the big shared folders (the 769 MB audio tokenizer) instead of
    copying them. Returns how: 'wsl-symlink', 'symlink', 'junction' or 'copy'."""
    if os.path.lexists(dst):
        return "existing"
    dst_wsl = wsl_parts(dst)
    if dst_wsl is not None and sys.platform == "win32":
        distro, dst_linux = dst_wsl
        src_linux = to_linux_path(src, distro)
        if src_linux:
            r = subprocess.run(["wsl.exe", "-d", distro, "--", "ln", "-s", src_linux, dst_linux],
                               capture_output=True)
            if r.returncode == 0:
                return "wsl-symlink"
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except OSError:
        pass
    if sys.platform == "win32" and dst_wsl is None and wsl_parts(src) is None:
        r = subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], capture_output=True)
        if r.returncode == 0:
            return "junction"
    shutil.copytree(src, dst)
    return "copy"


def prune_exports(export_dir: str, current: str, keep: int, log: Callable[[str], None] = print):
    """Keep the newest ``keep`` exports of this job. Only removes what an
    export wrote (per its marker); links are unlinked, never followed."""
    job_prefix = re.sub(r"_\d+$", "", current)
    exports = []
    for d in glob.glob(os.path.join(export_dir, "*", EXPORT_MARKER)):
        folder = os.path.dirname(d)
        base = os.path.basename(folder)
        if re.sub(r"_\d+$", "", base) == job_prefix:
            exports.append((os.path.getmtime(d), folder))
    exports.sort(reverse=True)
    for _, folder in exports[keep:]:
        try:
            with open(os.path.join(folder, EXPORT_MARKER), encoding="utf-8") as f:
                marker = json.load(f)
            for fname in marker.get("files", []):
                p = os.path.join(folder, fname)
                if os.path.isfile(p):
                    os.remove(p)
            for dname, how in marker.get("dirs", {}).items():
                p = os.path.join(folder, dname)
                if how == "wsl-symlink":
                    distro, linux = wsl_parts(p)
                    subprocess.run(["wsl.exe", "-d", distro, "--", "rm", "--", linux], capture_output=True)
                elif how in ("symlink", "junction"):
                    os.unlink(p) if how == "symlink" else os.rmdir(p)
                elif how == "copy":
                    shutil.rmtree(p)
            os.remove(os.path.join(folder, EXPORT_MARKER))
            os.rmdir(folder)
            log(f"omnivoice: removed old export {os.path.basename(folder)}")
        except Exception as e:
            log(f"omnivoice: could not remove old export {folder}: {e}")
