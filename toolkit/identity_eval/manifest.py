"""Run manifest and calibration fingerprint (spec §2.3, §5.3).

The calibration id hashes everything the step-0 anchors depend on: base model,
suites (prompts, seeds, sizes, renderer per slot), sampler settings, reference
images and scoring backends. A run whose fingerprint no longer matches its
saved anchors must not reuse them.
"""

import hashlib
import json
import os
from typing import Iterable, List

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint_model(name_or_path: str) -> dict:
    """Cheap fingerprint of a base model: size + mtime for local files (a full
    sha256 of a multi-GB checkpoint every run is too slow), the id for a hub
    repo."""
    if name_or_path and os.path.isfile(name_or_path):
        st = os.stat(name_or_path)
        return {"path": os.path.abspath(name_or_path), "size": st.st_size, "mtime": int(st.st_mtime)}
    if name_or_path and os.path.isdir(name_or_path):
        files = []
        for root, _, names in os.walk(name_or_path):
            for n in names:
                if n.endswith(('.safetensors', '.bin', '.json')):
                    p = os.path.join(root, n)
                    st = os.stat(p)
                    files.append([os.path.relpath(p, name_or_path), st.st_size, int(st.st_mtime)])
        return {"path": os.path.abspath(name_or_path), "files": sorted(files)}
    return {"repo": name_or_path}


def list_images(folders: Iterable[str]) -> List[str]:
    out = []
    for folder in folders:
        if folder and os.path.isdir(folder):
            out.extend(
                os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if f.lower().endswith(IMAGE_EXTS)
            )
    return out


def hash_images(paths: Iterable[str]) -> List[List[str]]:
    return [[os.path.basename(p), sha256_file(p)] for p in paths]


def calibration_id(fingerprint: dict) -> str:
    blob = json.dumps(fingerprint, sort_keys=True, separators=(',', ':'))
    return "sha256:" + hashlib.sha256(blob.encode('utf-8')).hexdigest()


def read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
