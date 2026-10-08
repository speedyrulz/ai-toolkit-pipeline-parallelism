"""Export trained OmniVoice LoRA files as merged models for ComfyUI.

The ComfyUI OmniVoice nodes cannot load LoRA files, so each LoRA is folded into
a copy of the base model and written as a model folder (named after the LoRA
file) into ComfyUI's models/omnivoice folder. This is what the trainer does
after each save when comfyui_export_dir is set; use this script for LoRAs that
were trained without it.

    python scripts/export_omnivoice_lora.py output/my_voice/my_voice.safetensors
        --dest \\\\wsl.localhost\\Ubuntu-24.04\\home\\me\\ComfyUI\\models\\omnivoice

Several LoRA files can be given at once. The base model is read from the
config.yaml next to the LoRA (model.name_or_path), else --base, else
k2-fsa/OmniVoice. Load the exports in ComfyUI with dtype fp32.
"""

import argparse
import importlib.util
import os
import sys

_EXPORT_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extensions_built_in",
                          "audio_models", "omnivoice", "export.py")


def _load_export_module():
    # loaded by path: importing the extension package would load the whole toolkit
    spec = importlib.util.spec_from_file_location("omnivoice_export", _EXPORT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_from_config(lora_path: str):
    cfg = os.path.join(os.path.dirname(os.path.abspath(lora_path)), "config.yaml")
    if not os.path.isfile(cfg):
        return None
    import yaml

    with open(cfg, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return data["config"]["process"][0]["model"]["name_or_path"] or None
    except (KeyError, IndexError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Export OmniVoice LoRAs as merged ComfyUI models")
    parser.add_argument("loras", nargs="+", help="LoRA .safetensors files")
    parser.add_argument("--dest", required=True, help="ComfyUI models/omnivoice folder")
    parser.add_argument("--base", default=None, help="base model (repo id or folder); default from config.yaml")
    parser.add_argument("--dtype", default="fp32", help="fp32 (default), fp16 or bf16")
    args = parser.parse_args()

    export = _load_export_module()
    try:
        from omnivoice.models.omnivoice import _resolve_model_path
    except ImportError:
        sys.exit("The omnivoice package is required: pip install omnivoice==0.2.1 --no-deps")
    dtype = export.DTYPES[args.dtype.lower()]

    for lora in args.loras:
        if not os.path.isfile(lora):
            sys.exit(f"not a file: {lora}")
        base_id = args.base or _base_from_config(lora) or "k2-fsa/OmniVoice"
        base_path = _resolve_model_path(base_id)
        print(f"merging {os.path.basename(lora)} into {base_id}")
        state = export.merge_lora_file(base_path, lora, dtype)
        export.write_export(state, lora, base_path, args.dest)


if __name__ == "__main__":
    main()
