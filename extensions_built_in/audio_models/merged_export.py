"""Merged-model export of audio-model LoRAs for ComfyUI.

The ComfyUI OmniVoice and Qwen3-TTS nodes cannot load LoRA files; they load a
model folder. An export is such a folder: the base model's files with the LoRA
folded into ``model.safetensors``, and the big shared sub-models (OmniVoice's
audio tokenizer, Qwen3-TTS's speech tokenizer) linked rather than copied.

Used by the trainers after each save (``comfyui_export_dir``) and, for LoRAs
that are already trained, by ``scripts/export_omnivoice_lora.py`` and
``scripts/export_qwen3_tts_lora.py``. Has no toolkit imports so the scripts
stay light.
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
# (defaults: OmniVoice)
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
def merge_lora_file(base_path: str, lora_path: str, dtype: torch.dtype = torch.float32,
                    skip_prefixes=()) -> dict:
    """Base model.safetensors with a saved LoRA file folded in. Keys are
    relative to the OmniVoice root (llm.layers.N.self_attn.q_proj...). PEFT
    format (lora_A/lora_B) carries no alpha, so its scale is 1; kohya format
    (lora_down/lora_up/.alpha) uses alpha / rank. Keys starting with one of
    ``skip_prefixes`` (non-LoRA data a model keeps in its LoRA file) are left
    for the caller."""
    from safetensors.torch import load_file

    base = load_file(os.path.join(base_path, "model.safetensors"))
    lora = load_file(lora_path)
    modules = set()
    for k in lora:
        if k.startswith(tuple(skip_prefixes)):
            continue
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


@torch.no_grad()
def merged_state_dict(model, network, dtype: torch.dtype = torch.float32,
                      base_weights_file: Optional[str] = None) -> dict:
    """``model``'s weights with every LoRA delta of the (live, trained)
    ``network`` folded in, cast to ``dtype``. Computed on fp32 stand-in copies
    of the wrapped layers so the live frozen weights are never merged in and
    out here. The base weights come from ``base_weights_file`` when given: the
    trainer's sample rounds merge the LoRA into the bf16 model and back out,
    and each round trip moves some live base weights by a bf16 step."""
    names = {id(m): n for n, m in model.named_modules()}
    dt = dtype
    base = {}
    if base_weights_file:
        from safetensors.torch import load_file

        base = load_file(base_weights_file)
    merged = {}
    for k, v in model.state_dict().items():
        v = base.get(k, v)
        merged[k] = (v.detach().to("cpu", dt) if v.is_floating_point() else v.detach().cpu()).clone()
    prev_mult = network.multiplier
    network.multiplier = 1.0
    network._update_torch_multiplier()
    try:
        for module in network.get_all_modules():
            org = module.org_module[0]
            name = names.get(id(org))
            if name is None:
                continue
            stand_in = torch.nn.Linear(org.in_features, org.out_features, bias=org.bias is not None,
                                       device=org.weight.device, dtype=torch.float32)
            stand_in.weight.copy_(base.get(f"{name}.weight", org.weight).float())
            if org.bias is not None:
                stand_in.bias.copy_(base.get(f"{name}.bias", org.bias).float())
            module.org_module[0] = stand_in
            prev_merged = getattr(module, "is_merged", None)
            try:
                module.merge_in(1.0)
            finally:
                module.org_module[0] = org
                if prev_merged is not None:
                    module.is_merged = prev_merged
            merged[f"{name}.weight"] = stand_in.weight.detach().to("cpu", dt)
    finally:
        network.multiplier = prev_mult
        network._update_torch_multiplier()
    return merged


def write_export(state: dict, lora_path: str, base_path: str, export_dir: str, keep: int = 0,
                 log: Callable[[str], None] = print, link_dirs=LINK_DIRS, skip_files=SKIP_FILES,
                 edit_files: Optional[dict] = None) -> str:
    """Write ``state`` as a model folder named after the LoRA file into
    ``export_dir``. ``edit_files`` maps a copied file's name to a function
    taking and returning its text (e.g. a config.json change). Returns the
    folder."""
    from safetensors.torch import save_file

    t0 = time.time()
    export_dir = _canonical(export_dir)
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
        if os.path.isfile(src) and entry not in skip_files and not entry.startswith("."):
            if edit_files and entry in edit_files:
                with open(src, encoding="utf-8") as f:
                    text = edit_files[entry](f.read())
                with open(os.path.join(dest, entry), "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                shutil.copy2(src, os.path.join(dest, entry))
            written.append(entry)
    linked = {}
    for d in link_dirs:
        src = os.path.join(base_path, d)
        if os.path.isdir(src):
            linked[d] = link_dir(src, os.path.join(dest, d))
    with open(os.path.join(dest, EXPORT_MARKER), "w", encoding="utf-8") as f:
        json.dump({"source_lora": os.path.abspath(lora_path), "files": written, "dirs": linked}, f, indent=2)
    log(f"exported merged model to {dest} ({time.time() - t0:.0f}s)")
    if keep > 0:
        prune_exports(export_dir, name, keep, log)
    return dest


def _canonical(path: str) -> str:
    """A mapped drive letter (or other alias) of a \\\\wsl.localhost share as
    its UNC path, so WSL destinations are recognised however they are named."""
    if sys.platform != "win32" or wsl_parts(path) is not None:
        return path
    try:
        real = os.path.realpath(path)
    except OSError:
        return path
    if real.startswith("\\\\?\\UNC\\"):
        real = "\\\\" + real[8:]
    elif real.startswith("\\\\?\\"):
        real = real[4:]
    return real if wsl_parts(real) is not None else path


def _local_fixed_drive(path: str) -> bool:
    """Junctions only work on a local NTFS disk, not on a network share
    (a junction made on the WSL share is an empty folder inside WSL)."""
    import ctypes

    drive = os.path.splitdrive(os.path.abspath(path))[0]
    if len(drive) != 2 or drive[1] != ":":
        return False
    return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == 3  # DRIVE_FIXED


def _link_works(src: str, dst: str, how: str) -> bool:
    probe = next((f for f in sorted(os.listdir(src)) if os.path.isfile(os.path.join(src, f))), None)
    if probe is None:
        return True
    if how == "wsl-symlink":
        distro, dst_linux = wsl_parts(dst)
        r = subprocess.run(["wsl.exe", "-d", distro, "--", "test", "-f", f"{dst_linux}/{probe}"], capture_output=True)
        return r.returncode == 0
    return os.path.isfile(os.path.join(dst, probe))


def _unlink(dst: str, how: str):
    if how == "wsl-symlink":
        distro, dst_linux = wsl_parts(dst)
        subprocess.run(["wsl.exe", "-d", distro, "--", "rm", "--", dst_linux], capture_output=True)
    elif how == "junction":
        os.rmdir(dst)
    else:
        os.unlink(dst)


def link_dir(src: str, dst: str) -> str:
    """Link the big shared folders (the audio / speech tokenizer) instead of
    copying them. Every link is checked from where the model will be loaded;
    one that does not show the files is removed and the folder copied.
    Returns how: 'wsl-symlink', 'symlink', 'junction', 'copy' or 'existing'."""
    dst = _canonical(dst)
    if os.path.lexists(dst):
        if os.path.isdir(dst) and not os.listdir(dst) and not os.path.islink(dst):
            os.rmdir(dst)  # an empty leftover (e.g. a junction made on the WSL share)
        else:
            return "existing"
    dst_wsl = wsl_parts(dst)
    attempts = []
    if dst_wsl is not None and sys.platform == "win32":
        distro, dst_linux = dst_wsl
        src_linux = to_linux_path(src, distro)
        if src_linux:
            attempts.append(("wsl-symlink", ["wsl.exe", "-d", distro, "--", "ln", "-s", src_linux, dst_linux]))
    # Windows links on a share (a mapped WSL drive among them) can look fine
    # from Windows and empty from Linux: only make them on a local disk
    if dst_wsl is None and (sys.platform != "win32" or _local_fixed_drive(dst)):
        attempts.append(("symlink", None))
        if sys.platform == "win32" and wsl_parts(src) is None:
            # mklink reads a forward slash as a switch
            attempts.append(("junction", ["cmd", "/c", "mklink", "/J", os.path.normpath(dst), os.path.normpath(src)]))
    for how, cmd in attempts:
        try:
            if cmd is None:
                os.symlink(src, dst, target_is_directory=True)
            elif subprocess.run(cmd, capture_output=True).returncode != 0:
                continue
        except OSError:
            continue
        if _link_works(src, dst, how):
            return how
        _unlink(dst, how)
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
            log(f"removed old export {os.path.basename(folder)}")
        except Exception as e:
            log(f"could not remove old export {folder}: {e}")
