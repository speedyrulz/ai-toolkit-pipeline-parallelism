"""Sample helper: renders part of each sample round on a second gpu.

    python -m toolkit.identity_eval.sample_helper --spec <spec.json>

Started by the trainer with CUDA_VISIBLE_DEVICES pinned to the helper card, so
the model lives on cuda:0 here. It loads the job's base model once through the
regular model loader, and each round applies the trainer's LoRA snapshot as
forward hooks (toolkit/inference_lora.py), at each image's network multiplier.

Commands:
  render  {items: [GenerateImageConfig kwargs], lora_path, sampler}
  offload / onload   move the model to cpu and back around gpu scoring
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    from toolkit.win_console import suppress_child_consoles
    suppress_child_consoles()
    os.environ["DISABLE_TELEMETRY"] = "YES"

    import torch
    from types import SimpleNamespace
    from toolkit.config_modules import GenerateImageConfig, ModelConfig
    from toolkit.identity_eval.proc import serve
    from toolkit.inference_lora import InferenceLoRA
    from toolkit.sampler import get_sampler
    from toolkit.util.get_model import get_model_class
    from toolkit.basic import flush

    model_config = ModelConfig(**spec["model"])
    ModelClass = get_model_class(model_config)
    if hasattr(ModelClass, "get_train_scheduler"):
        noise_scheduler = ModelClass.get_train_scheduler()
    else:
        arch = "sd"
        if model_config.is_pixart:
            arch = "pixart"
        if model_config.is_flux:
            arch = "flux"
        if model_config.is_lumina2:
            arch = "lumina2"
        noise_scheduler = get_sampler(
            spec["noise_scheduler"],
            {"prediction_type": "v_prediction" if model_config.is_v_pred else "epsilon"},
            arch=arch,
        )
    print(f"sample helper: loading {model_config.name_or_path} on cuda:0", flush=True)
    sd = ModelClass(
        device="cuda:0",
        model_config=model_config,
        dtype=spec["dtype"],
        noise_scheduler=noise_scheduler,
    )
    with torch.no_grad():
        sd.load_model()
    state = {"lora": None, "lora_path": None, "lora_mtime": None, "offloaded": False}

    def holder():
        return SimpleNamespace(
            model=sd.unet_unwrapped,
            text_encoder=sd.text_encoder,
            convert_lora_weights_before_load=getattr(sd, "convert_lora_weights_before_load", None),
        )

    def get_lora(path):
        if not path:
            return None
        mtime = os.path.getmtime(path)
        if state["lora"] is not None and state["lora_path"] == path and state["lora_mtime"] == mtime:
            return state["lora"]
        lora = InferenceLoRA(path, name="trainer").load(holder())
        if not lora.entries:
            raise RuntimeError(f"LoRA snapshot {path} matched no modules of the helper model")
        state.update(lora=lora, lora_path=path, lora_mtime=mtime)
        return lora

    def render(items, lora_path=None, sampler=None):
        if state["offloaded"]:
            raise RuntimeError("render while offloaded")
        lora = get_lora(lora_path)
        groups = {}
        for item in items:
            groups.setdefault(float(item.get("network_multiplier", 1.0)), []).append(item)
        written = []
        with torch.no_grad():
            for mult, group in groups.items():
                configs = [GenerateImageConfig(**item) for item in group]
                use_lora = lora is not None and mult != 0.0
                if use_lora:
                    lora.strength = mult
                    lora.attach()
                try:
                    sd.generate_images(configs, sampler=sampler)
                finally:
                    if use_lora:
                        lora.detach()
                written.extend(c.output_path for c in configs)
        flush()
        return {"written": written}

    def offload():
        if not state["offloaded"]:
            sd.set_device_state_preset("unload")
            state["offloaded"] = True
            flush()
        return {}

    def onload():
        if state["offloaded"]:
            sd.restore_device_state()
            state["offloaded"] = False
        return {}

    serve({"render": render, "offload": offload, "onload": onload})


if __name__ == "__main__":
    main()
