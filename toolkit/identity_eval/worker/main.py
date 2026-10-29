"""Identity eval scoring worker (runs in the evaluator environment).

    python toolkit/identity_eval/worker/main.py --role face|body --device cpu|cuda
        [--backend auto|stub] [--models '<json>']

The face worker also scores the flex suite (prompt adherence).
The trainer pins the process to one card with CUDA_VISIBLE_DEVICES, so a gpu
worker always sees it as cuda:0. The worker only extracts raw features; all
calibration math runs trainer-side in toolkit/identity_eval/scoring.py.

Commands (see toolkit/identity_eval/proc.py for the wire format):
  prepare  {references, cache_dir}  build reference templates; returns, per
           feature, the leave-one-out raw values that set the 100 anchor
  score    {images: [{slot, path, modality, prompt?}]} raw features per
           image, or a status and reason when the image is unusable
  to_gpu / to_cpu                   move models between rounds
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # toolkit/identity_eval, for proc.py
sys.path.insert(0, HERE)

from proc import serve  # noqa: E402


def build_backend(role: str, device: str, backend: str, models: dict):
    # face v1 is real; body stays on the stub until P2
    if backend == "auto" and role == "face":
        from face import FaceBackend
        return FaceBackend(role, device, models)
    from stub import StubBackend
    return StubBackend(role, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["face", "body"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--backend", default="auto", choices=["auto", "stub"])
    parser.add_argument("--models", default="{}", help="model overrides as json")
    args = parser.parse_args()

    backend = build_backend(args.role, args.device, args.backend, json.loads(args.models))
    print(f"identity eval worker: role={args.role} device={args.device} "
          f"backend={backend.name} metric_version={backend.metric_version}", flush=True)

    serve({
        "info": lambda: {"backend": backend.name, "metric_version": backend.metric_version,
                         "features": backend.features,
                         "weights": getattr(backend, "weights", {f: 1.0 for f in backend.features}),
                         "models": getattr(backend, "model_ids", {})},
        "prepare": backend.prepare,
        "score": backend.score,
        "to_gpu": backend.to_gpu,
        "to_cpu": backend.to_cpu,
    })


if __name__ == "__main__":
    main()
