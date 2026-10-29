"""Trainer-side handle for the sample helper process (sample_helper.py)."""

import json
import os
import sys
from typing import List, Optional

from toolkit.print import print_acc
from toolkit.identity_eval.coordinator import gpu_env
from toolkit.identity_eval.proc import ChildError, JsonProcess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SampleHelper:
    def __init__(self, gpu: int, work_dir: str, model_conf: dict, dtype: str, noise_scheduler: str):
        self.gpu = gpu
        self.work_dir = work_dir
        self.spec = {"model": model_conf, "dtype": dtype, "noise_scheduler": noise_scheduler}
        self.proc: Optional[JsonProcess] = None
        self.offloaded = False
        self._render_id: Optional[int] = None

    def start(self) -> bool:
        os.makedirs(self.work_dir, exist_ok=True)
        spec_path = os.path.join(self.work_dir, "sample_helper_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(self.spec, f, indent=2, default=str)
        print_acc(f"Starting sample helper on gpu {self.gpu} (loads its own copy of the base model)")
        self.proc = JsonProcess(
            "sample helper",
            [sys.executable, "-u", "-m", "toolkit.identity_eval.sample_helper", "--spec", spec_path],
            os.path.join(self.work_dir, "sample_helper.log"),
            env=gpu_env(self.gpu),
            cwd=REPO,
        )
        try:
            self.proc.start()
        except ChildError as e:
            print_acc(f"Sample helper unavailable, sampling stays on one gpu: {e}")
            self.proc = None
            return False
        return True

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.alive

    def render_async(self, items: List[dict], lora_path: Optional[str], sampler: str,
                     sample_loras: Optional[List[dict]] = None):
        self._render_id = self.proc.send("render", items=items, lora_path=lora_path, sampler=sampler,
                                         sample_loras=sample_loras or [])

    def render_wait(self):
        req, self._render_id = self._render_id, None
        if req is not None:
            self.proc.wait(req)

    def offload(self):
        if self.alive and not self.offloaded:
            self.proc.request("offload", timeout=600)
            self.offloaded = True

    def onload(self):
        if self.alive and self.offloaded:
            self.proc.request("onload", timeout=600)
            self.offloaded = False

    def close(self):
        if self.proc is not None:
            self.proc.close()
            self.proc = None
