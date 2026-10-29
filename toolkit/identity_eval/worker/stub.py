"""Placeholder scoring backend for P0 (metric_version 0.0.0-stub).

Not an identity measure. It compares small grayscale thumbnails by cosine
similarity against a reference template so the whole pipeline (reference
templates, leave-one-out targets, paired baselines, worker protocol, gpu hand
off) can run end to end before the real models land. Needs numpy and Pillow.
"""

import json
import os
from typing import List

import numpy as np
from PIL import Image

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


def list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    )


class StubBackend:
    name = "stub"
    metric_version = "0.0.0-stub"

    def __init__(self, role: str, device: str):
        self.role = role
        self.device = device
        self.feature = f"stub_{role}"
        self.features = [self.feature]
        self.refs: List[np.ndarray] = []
        self.template = None

    def _vector(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        w, h = img.size
        if self.role == "face":
            # upper-centre crop, where a portrait's face usually sits
            img = img.crop((w * 0.25, 0, w * 0.75, h * 0.5)).resize((32, 32))
        else:
            img = img.resize((24, 48))
        v = np.asarray(img, dtype=np.float32).ravel()
        v = v - v.mean()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / n) if n > 0 else 0.0

    def prepare(self, references: List[str], cache_dir: str, **_):
        paths = []
        for r in references:
            paths.extend(list_images(r) if os.path.isdir(r) else [r])
        excluded = []
        self.refs = []
        for p in paths:
            try:
                self.refs.append(self._vector(p))
            except Exception as e:
                excluded.append({"path": p, "reason": f"unreadable: {e}"})
        if len(self.refs) < 2:
            raise RuntimeError(f"need at least 2 usable reference images, found {len(self.refs)}")
        stack = np.stack(self.refs)
        self.template = stack.mean(axis=0)
        loo = []
        for i in range(len(self.refs)):
            others = np.delete(stack, i, axis=0).mean(axis=0)
            loo.append(self._cos(self.refs[i], others))
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, f"{self.feature}_template.npy"), self.template)
        with open(os.path.join(cache_dir, f"{self.feature}_refs.json"), "w", encoding="utf-8") as f:
            json.dump({"references": paths, "excluded": excluded}, f, indent=2)
        return {
            "metric_version": self.metric_version,
            "reference_count": len(self.refs),
            "excluded": excluded,
            "features": {self.feature: {"loo": loo}},
        }

    def score(self, images: List[dict], **_):
        if self.template is None:
            raise RuntimeError("prepare has not run")
        results = []
        for item in images:
            try:
                raw = self._cos(self._vector(item["path"]), self.template)
                results.append({"slot": item["slot"], "status": "VALID", "raws": {self.feature: raw}})
            except Exception as e:
                results.append({"slot": item["slot"], "status": "FAILED", "reason": str(e), "raws": {}})
        return {"results": results}

    def to_gpu(self, **_):
        return {}

    def to_cpu(self, **_):
        return {}
