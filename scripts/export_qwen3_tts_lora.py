"""Export trained Qwen3-TTS LoRA files as merged custom-voice models for ComfyUI.

The ComfyUI-Qwen3-TTS nodes cannot load LoRA files, so each LoRA is folded into
a copy of the Base model, the trained voice is stored as a named speaker, and
the result is written as a model folder (named after the LoRA file) into
ComfyUI's models/Qwen3-TTS folder. This is what the trainer does after each
save when comfyui_export_dir is set; use this script for LoRAs that were
trained without it.

    python scripts/export_qwen3_tts_lora.py output/my_voice/my_voice.safetensors
        --dest \\\\wsl.localhost\\Ubuntu-24.04\\home\\me\\ComfyUI\\models\\Qwen3-TTS

In ComfyUI: Qwen3-TTS Loader with local_model_path = the exported folder and
precision fp32, then the Custom Voice node with custom_speaker_name = the
speaker name (--speaker-name, else the model_kwargs.speaker_name in the
config.yaml next to the LoRA, else the job name). The base model is read from
that config.yaml (model.name_or_path), else --base.
"""

import argparse
import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPORT_PY = os.path.join(_HERE, "..", "extensions_built_in", "audio_models", "merged_export.py")
SPEAKER_KEY = "qwen3_tts.speaker_embedding"
SPEAKER_ROW = 3000


def _load_export_module():
    # loaded by path: importing the extension package would load the whole toolkit
    spec = importlib.util.spec_from_file_location("audio_merged_export", _EXPORT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job_model_config(lora_path: str) -> dict:
    cfg = os.path.join(os.path.dirname(os.path.abspath(lora_path)), "config.yaml")
    if not os.path.isfile(cfg):
        return {}
    import yaml

    with open(cfg, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return data["config"]["process"][0]["model"] or {}
    except (KeyError, IndexError, TypeError):
        return {}


def custom_voice_config(config_text: str, speaker_name: str) -> str:
    import json

    cfg = json.loads(config_text)
    key = speaker_name.lower()
    cfg["tts_model_type"] = "custom_voice"
    cfg["talker_config"]["spk_id"] = {key: SPEAKER_ROW}
    cfg["talker_config"]["spk_is_dialect"] = {key: False}
    return json.dumps(cfg, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS LoRAs as merged ComfyUI custom-voice models")
    parser.add_argument("loras", nargs="+", help="LoRA .safetensors files")
    parser.add_argument("--dest", required=True, help="ComfyUI models/Qwen3-TTS folder")
    parser.add_argument("--base", default=None, help="Base model (repo id or folder); default from config.yaml")
    parser.add_argument("--speaker-name", default=None, help="voice name in ComfyUI; default from config.yaml")
    parser.add_argument("--dtype", default="fp32", help="fp32 (default), fp16 or bf16")
    args = parser.parse_args()

    export = _load_export_module()
    dtype = export.DTYPES[args.dtype.lower()]
    from safetensors import safe_open

    for lora in args.loras:
        if not os.path.isfile(lora):
            sys.exit(f"not a file: {lora}")
        job = _job_model_config(lora)
        base_id = args.base or job.get("name_or_path") or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        if os.path.isdir(base_id):
            base_path = base_id
        else:
            from huggingface_hub import snapshot_download

            base_path = snapshot_download(base_id)
        name = (args.speaker_name or (job.get("model_kwargs") or {}).get("speaker_name")
                or re.sub(r"_\d+$", "", os.path.splitext(os.path.basename(lora))[0])).lower()
        with safe_open(lora, "pt") as f:
            if SPEAKER_KEY not in f.keys():
                sys.exit(f"{os.path.basename(lora)} has no speaker embedding ({SPEAKER_KEY}); "
                         "was it trained with arch qwen3_tts?")
            speaker = f.get_tensor(SPEAKER_KEY)
        print(f"merging {os.path.basename(lora)} into {base_id} as speaker '{name}'")
        state = export.merge_lora_file(base_path, lora, dtype, skip_prefixes=(SPEAKER_KEY,))
        key = "talker.model.codec_embedding.weight"
        state[key][SPEAKER_ROW] = speaker[0].to(state[key].dtype)
        export.write_export(state, lora, base_path, args.dest, link_dirs=("speech_tokenizer",),
                            edit_files={"config.json": lambda t, n=name: custom_voice_config(t, n)})


if __name__ == "__main__":
    main()
