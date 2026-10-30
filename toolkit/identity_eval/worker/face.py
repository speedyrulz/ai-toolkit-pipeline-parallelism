"""Face v1 and flex guard backend (metric_version 1.0.0-face-v1).

Face: SCRFD detection -> main-subject selection -> 5-point alignment to the
ArcFace 112x112 template -> two independent embeddings, each its own feature:
  face_cvl  CVLFace AdaFace ViT-B KP-RPE (WebFace12M), keypoints from the DFA aligner
  face_arc  InsightFace AntelopeV2 ArcFace R100 (glintr100)
Raw value = cosine to the reference prototype of the face's view bin (front /
three_quarter / profile from a 5-point yaw estimate), or to the all-view
prototype when that bin has fewer than MIN_BIN_REFS references. The
leave-one-out target uses the same rule with the evaluated reference removed.

Flex: SigLIP2 text-image cosine between a flex image and its prompt (trigger
replaced by "a person"); the trainer reports the change from step 0.

Model files:
  <face_pack_dir>/scrfd_10g_bnkps.onnx, glintr100.onnx   (antelopev2 pack, fetched
      from the official insightface release when missing)
  CVLFace, DFA aligner and SigLIP2 from Hugging Face (huggingface_hub cache).
The CVLFace repos are imported as plain modules and loaded from safetensors;
their transformers wrappers and torch pickles are not used.
"""

import json
import math
import os
import sys
import types
import zipfile
from typing import Dict, List, Optional

import cv2
import numpy as np

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

DEFAULT_MODELS = {
    "face_pack_dir": os.path.join(os.path.expanduser("~"), ".insightface", "models", "antelopev2"),
    "face_pack_url": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
    "cvlface_repo": "minchul/cvlface_adaface_vit_base_kprpe_webface12m",
    "aligner_repo": "minchul/cvlface_DFA_mobilenet",
    "flex_model": "google/siglip2-base-patch16-384",
}

# ArcFace 112x112 five-point template (eyes, nose, mouth corners)
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

DET_SIZE = 640
DET_THRESHOLD = 0.5
PAD_RATIO = 1 / 3         # border for the second detection pass
NMS_THRESHOLD = 0.4
MIN_FACE_PX = 48          # shorter bbox side; smaller faces are NO_FACE
AMBIGUOUS_RATIO = 0.6     # a second face this large (area) makes the subject ambiguous
MIN_BIN_REFS = 2
FRONT_MAX_YAW = 20.0
THREE_QUARTER_MAX_YAW = 55.0
FEATURE_WEIGHTS = {"face_cvl": 0.6, "face_arc": 0.4}


def list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))


def read_bgr(path: str) -> np.ndarray:
    # imdecode handles non-ascii windows paths that imread does not
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("unreadable image")
    return img


def l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


# ----------------------------------------------------------------------
# view estimate
# ----------------------------------------------------------------------
def estimate_yaw(kps: np.ndarray) -> float:
    """Approximate yaw in degrees from the five SCRFD points: horizontal nose
    offset from the eye midpoint, in a roll-corrected frame, relative to the
    projected eye distance (offset/iod ~ k*tan(yaw), k ~ 0.55)."""
    le, re, nose = kps[0], kps[1], kps[2]
    d = re - le
    iod = float(np.hypot(*d))
    if iod < 1e-6:
        return 90.0
    c, s = d[0] / iod, d[1] / iod
    off = nose - (le + re) / 2.0
    x = off[0] * c + off[1] * s
    return math.degrees(math.atan2(x / iod, 0.55))


def yaw_bin(yaw: float) -> str:
    a = abs(yaw)
    if a < FRONT_MAX_YAW:
        return "front"
    if a < THREE_QUARTER_MAX_YAW:
        return "three_quarter"
    return "profile"


# ----------------------------------------------------------------------
# SCRFD detector (onnx)
# ----------------------------------------------------------------------
def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thresh]
    return keep


def _onnx_session(path: str, providers):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # glintr100 declares batch 1 and warns on every batch
    return ort.InferenceSession(path, providers=providers)


class Scrfd:
    strides = (8, 16, 32)
    anchors = 2

    def __init__(self, path: str, providers):
        self.sess = _onnx_session(path, providers)
        self.input_name = self.sess.get_inputs()[0].name
        if len(self.sess.get_outputs()) != 9:
            raise RuntimeError(f"{path} is not a SCRFD model with keypoints")

    def detect(self, img: np.ndarray):
        """[(box[4], score, kps[5,2])] in original image pixels."""
        h, w = img.shape[:2]
        scale = DET_SIZE / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (DET_SIZE, DET_SIZE), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.input_name: blob})
        all_boxes, all_scores, all_kps = [], [], []
        for i, stride in enumerate(self.strides):
            scores = outs[i].reshape(-1)
            dist = outs[i + 3].reshape(-1, 4) * stride
            kdist = outs[i + 6].reshape(-1, 10) * stride
            fs = DET_SIZE // stride
            ys, xs = np.mgrid[:fs, :fs]
            centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            centers = np.repeat(centers, self.anchors, axis=0)
            pos = np.where(scores >= DET_THRESHOLD)[0]
            if not len(pos):
                continue
            c = centers[pos]
            d = dist[pos]
            boxes = np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=1)
            k = kdist[pos].reshape(-1, 5, 2) + c[:, None, :]
            all_boxes.append(boxes)
            all_scores.append(scores[pos])
            all_kps.append(k)
        if not all_boxes:
            return []
        boxes = np.concatenate(all_boxes) / scale
        scores = np.concatenate(all_scores)
        kps = np.concatenate(all_kps) / scale
        return [(boxes[i], float(scores[i]), kps[i]) for i in _nms(boxes, scores, NMS_THRESHOLD)]

    def detect_padded(self, img: np.ndarray):
        """detect(), retried on a black border of PAD_RATIO per side when it
        finds nothing: SCRFD misses close-ups whose face fills the frame and
        runs off the edge, which the border brings back into its anchor range."""
        faces = self.detect(img)
        if faces:
            return faces
        h, w = img.shape[:2]
        py, px = int(h * PAD_RATIO), int(w * PAD_RATIO)
        padded = cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=0)
        off = np.array([px, py], dtype=np.float32)
        return [(box - np.tile(off, 2), score, kps - off) for box, score, kps in self.detect(padded)]


def select_subject(faces) -> tuple:
    """(face or None, reason). Largest face wins; a second face of comparable
    size makes the image ambiguous."""
    if not faces:
        return None, "no face detected"
    areas = [(f[0][2] - f[0][0]) * (f[0][3] - f[0][1]) for f in faces]
    order = np.argsort(areas)[::-1]
    main = faces[order[0]]
    if len(faces) > 1 and areas[order[1]] >= AMBIGUOUS_RATIO * areas[order[0]]:
        return None, f"ambiguous: {len(faces)} faces of similar size"
    box = main[0]
    if min(box[2] - box[0], box[3] - box[1]) < MIN_FACE_PX:
        return None, f"face too small ({int(min(box[2] - box[0], box[3] - box[1]))}px)"
    return main, None


def align(img: np.ndarray, kps: np.ndarray) -> np.ndarray:
    m, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if m is None:
        raise ValueError("alignment failed")
    return cv2.warpAffine(img, m, (112, 112), borderValue=0.0)


class ArcFace:
    def __init__(self, path: str, providers):
        self.sess = _onnx_session(path, providers)
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        blob = cv2.dnn.blobFromImages(crops, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        return l2n(self.sess.run(None, {self.input_name: blob})[0])


# ----------------------------------------------------------------------
# CVLFace KP-RPE (torch)
# ----------------------------------------------------------------------
def _import_repo(repo_dir: str, package: str):
    sys.path.insert(0, repo_dir)
    try:
        mod = __import__(package)
    finally:
        sys.path.remove(repo_dir)
    if not os.path.abspath(mod.__file__).startswith(os.path.abspath(repo_dir)):
        raise RuntimeError(f"package {package!r} resolved outside {repo_dir}")
    return mod


def _load_prefixed(module, path: str, prefix: str = "model."):
    from safetensors.torch import load_file
    sd = load_file(path)
    sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    module.load_state_dict(sd, strict=True)


class CvlFace:
    def __init__(self, repo_dir: str, aligner_dir: str):
        import torch
        import yaml
        from easydict import EasyDict
        # the repo tries to compile and install a C++ op (setup.py install, then
        # exit) when it is missing; register it as absent so the pure torch path runs
        stub = types.ModuleType("models.vit_kprpe.RPE.rpe_ops.rpe_index")
        stub.RPEIndexFunction = None
        sys.modules[stub.__name__] = stub
        models = _import_repo(repo_dir, "models")
        aligners = _import_repo(aligner_dir, "aligners")
        with open(os.path.join(repo_dir, "pretrained_model", "model.yaml"), encoding="utf-8") as f:
            self.model = models.get_model(EasyDict(yaml.safe_load(f)))
        _load_prefixed(self.model, os.path.join(repo_dir, "model.safetensors"))
        with open(os.path.join(aligner_dir, "pretrained_model", "model.yaml"), encoding="utf-8") as f:
            self.aligner = aligners.get_aligner(EasyDict(yaml.safe_load(f)))
        _load_prefixed(self.aligner, os.path.join(aligner_dir, "model.safetensors"))
        self.model.eval()
        self.aligner.eval()
        self.torch = torch
        self.device = "cpu"

    def to(self, device: str):
        self.model.to(device)
        self.aligner.to(device)
        self.device = device

    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        torch = self.torch
        rgb = np.stack([c[:, :, ::-1] for c in crops]).astype(np.float32)
        x = torch.from_numpy(rgb).permute(0, 3, 1, 2).div(127.5).sub(1.0).to(self.device)
        with torch.no_grad():
            _, ldmks, _, _, _, _ = self.aligner(x)
            out = self.model(x, ldmks)
        return l2n(out.float().cpu().numpy())


# ----------------------------------------------------------------------
# SigLIP2 flex guard
# ----------------------------------------------------------------------
class Siglip:
    def __init__(self, path: str):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.model = AutoModel.from_pretrained(path, dtype=torch.float32).eval()
        self.processor = AutoProcessor.from_pretrained(path)
        self.device = "cpu"

    def to(self, device: str):
        self.model.to(device, dtype=self.torch.float16 if device == "cuda" else self.torch.float32)
        self.device = device

    @staticmethod
    def _features(out):
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def similarity(self, images: List[np.ndarray], prompts: List[str]) -> List[float]:
        torch = self.torch
        from PIL import Image
        pil = [Image.fromarray(i[:, :, ::-1]) for i in images]
        dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            px = self.processor(images=pil, return_tensors="pt")["pixel_values"].to(self.device, dtype)
            tok = self.processor(text=prompts, padding="max_length", max_length=64, truncation=True,
                                 return_tensors="pt").to(self.device)
            iv = self._features(self.model.get_image_features(pixel_values=px)).float()
            tv = self._features(self.model.get_text_features(**tok)).float()
        iv = torch.nn.functional.normalize(iv, dim=-1)
        tv = torch.nn.functional.normalize(tv, dim=-1)
        return (iv * tv).sum(-1).cpu().tolist()


# ----------------------------------------------------------------------
# model files
# ----------------------------------------------------------------------
def ensure_face_pack(pack_dir: str, url: str) -> Dict[str, str]:
    need = {"det": "scrfd_10g_bnkps.onnx", "rec": "glintr100.onnx"}
    paths = {k: os.path.join(pack_dir, v) for k, v in need.items()}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    os.makedirs(pack_dir, exist_ok=True)
    zip_path = pack_dir.rstrip("\\/") + ".zip"
    if not os.path.exists(zip_path):
        import urllib.request
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, zip_path + ".part")
        os.replace(zip_path + ".part", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in need.values():
                with z.open(name) as src, open(os.path.join(pack_dir, base), "wb") as dst:
                    dst.write(src.read())
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"face model pack incomplete: {missing}")
    return paths


def hf_snapshot(repo: str, patterns=None) -> str:
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=patterns)


# ----------------------------------------------------------------------
# backend
# ----------------------------------------------------------------------
class FaceBackend:
    name = "face-v1"
    metric_version = "1.0.1-face-v1"
    features = list(FEATURE_WEIGHTS)
    weights = dict(FEATURE_WEIGHTS)

    def __init__(self, role: str, device: str, models: Optional[dict] = None):
        import torch  # loads the CUDA runtime that onnxruntime's CUDA provider uses
        self.role = role
        self.device = device
        self.models = dict(DEFAULT_MODELS, **(models or {}))
        self.pack = ensure_face_pack(self.models["face_pack_dir"], self.models["face_pack_url"])
        cvl_dir = hf_snapshot(self.models["cvlface_repo"], ["*.py", "*.yaml", "*.json", "model.safetensors"])
        aln_dir = hf_snapshot(self.models["aligner_repo"], ["*.py", "*.yaml", "*.json", "model.safetensors"])
        flex_dir = hf_snapshot(self.models["flex_model"])
        self.cvl = CvlFace(cvl_dir, aln_dir)
        self.siglip = Siglip(flex_dir)
        self.model_ids = {
            "face_pack": os.path.basename(self.models["face_pack_dir"].rstrip("\\/")),
            "cvlface": self.models["cvlface_repo"],
            "aligner": self.models["aligner_repo"],
            "flex": self.models["flex_model"],
        }
        self.det = self.arc = None
        self._on_gpu = False
        self._open_onnx(gpu=False)
        self.refs: List[dict] = []

    # -- device handling --------------------------------------------------
    def _open_onnx(self, gpu: bool):
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"] if gpu \
            else ["CPUExecutionProvider"]
        self.det = Scrfd(self.pack["det"], providers)
        self.arc = ArcFace(self.pack["rec"], providers)

    def to_gpu(self, **_):
        if self.device != "cuda" or self._on_gpu:
            return {}
        self._open_onnx(gpu=True)
        self.cvl.to("cuda")
        self.siglip.to("cuda")
        self._on_gpu = True
        return {"providers": self.det.sess.get_providers()}

    def to_cpu(self, **_):
        if not self._on_gpu:
            return {}
        self._open_onnx(gpu=False)
        self.cvl.to("cpu")
        self.siglip.to("cpu")
        self.cvl.torch.cuda.empty_cache()
        self._on_gpu = False
        return {}

    # -- face features ----------------------------------------------------
    def _face(self, path: str) -> dict:
        img = read_bgr(path)
        face, reason = select_subject(self.det.detect_padded(img))
        if face is None:
            return {"status": "NO_FACE", "reason": reason}
        box, score, kps = face
        crop = align(img, kps)
        yaw = estimate_yaw(kps)
        return {
            "status": "VALID",
            "crop": crop,
            "yaw": round(yaw, 1),
            "view": yaw_bin(yaw),
            "det_score": round(score, 3),
            "face_px": int(min(box[2] - box[0], box[3] - box[1])),
        }

    def _embed(self, crops: List[np.ndarray]) -> Dict[str, np.ndarray]:
        out = {"face_cvl": [], "face_arc": []}
        for i in range(0, len(crops), 16):
            chunk = crops[i:i + 16]
            out["face_cvl"].append(self.cvl.embed(chunk))
            out["face_arc"].append(self.arc.embed(chunk))
        return {k: np.concatenate(v) for k, v in out.items()}

    def _pool(self, view: str, exclude: Optional[int] = None) -> List[int]:
        """Reference indices compared with a face of this view."""
        idx = [i for i in range(len(self.refs)) if i != exclude]
        same = [i for i in idx if self.refs[i]["view"] == view]
        return same if len(same) >= MIN_BIN_REFS else idx

    def _raw(self, feat: str, emb: np.ndarray, view: str, exclude: Optional[int] = None):
        pool = self._pool(view, exclude)
        allv = [i for i in range(len(self.refs)) if i != exclude]
        proto = l2n(self.ref_emb[feat][pool].mean(axis=0))
        xview = l2n(self.ref_emb[feat][allv].mean(axis=0))
        return float(emb @ proto), float(emb @ xview), len(pool)

    def prepare(self, references: List[str], cache_dir: str, **_):
        paths = []
        for r in references:
            paths.extend(list_images(r) if os.path.isdir(r) else [r])
        excluded, faces = [], []
        for p in paths:
            try:
                f = self._face(p)
            except Exception as e:
                f = {"status": "FAILED", "reason": f"unreadable: {e}"}
            if f["status"] != "VALID":
                excluded.append({"path": p, "reason": f["reason"]})
            else:
                f["path"] = p
                faces.append(f)
        if len(faces) < 3:
            raise RuntimeError(f"need at least 3 reference images with a usable face, found {len(faces)} "
                               f"({len(excluded)} excluded)")
        emb = self._embed([f["crop"] for f in faces])
        self.refs = [{k: v for k, v in f.items() if k != "crop"} for f in faces]
        self.ref_emb = emb
        loo = {feat: [] for feat in self.features}
        for i, ref in enumerate(self.refs):
            for feat in self.features:
                loo[feat].append(self._raw(feat, emb[feat][i], ref["view"], exclude=i)[0])
        os.makedirs(cache_dir, exist_ok=True)
        for feat in self.features:
            np.save(os.path.join(cache_dir, f"{feat}_embeddings.npy"), emb[feat])
        views = {}
        for r in self.refs:
            views[r["view"]] = views.get(r["view"], 0) + 1
        with open(os.path.join(cache_dir, "face_refs.json"), "w", encoding="utf-8") as f:
            json.dump({"references": self.refs, "excluded": excluded, "views": views,
                       "models": self.model_ids}, f, indent=2)
        print(f"face references: {len(self.refs)} usable {views}, {len(excluded)} excluded", flush=True)
        return {
            "metric_version": self.metric_version,
            "reference_count": len(self.refs),
            "excluded": excluded,
            "views": views,
            "features": {feat: {"loo": loo[feat]} for feat in self.features},
        }

    def _score_faces(self, items: List[dict]) -> List[dict]:
        results, valid = [], []
        for item in items:
            try:
                f = self._face(item["path"])
            except Exception as e:
                f = {"status": "FAILED", "reason": str(e)}
            res = {"slot": item["slot"], "status": f["status"], "reason": f.get("reason"), "raws": {}}
            if f["status"] == "VALID":
                res["extra"] = {k: f[k] for k in ("yaw", "view", "det_score", "face_px")}
                valid.append((res, f))
            results.append(res)
        if valid:
            emb = self._embed([f["crop"] for _, f in valid])
            for n, (res, f) in enumerate(valid):
                for feat in self.features:
                    raw, xview, pool = self._raw(feat, emb[feat][n], f["view"])
                    res["raws"][feat] = raw
                    res["extra"][f"{feat}_xview"] = round(xview, 4)
                res["extra"]["pool"] = pool
        return results

    def _score_flex(self, items: List[dict]) -> List[dict]:
        results, imgs, prompts, ok = [], [], [], []
        for item in items:
            res = {"slot": item["slot"], "status": "VALID", "reason": None, "raws": {}}
            try:
                imgs.append(read_bgr(item["path"]))
                prompts.append(item.get("prompt") or "")
                ok.append(res)
            except Exception as e:
                res.update(status="FAILED", reason=str(e))
            results.append(res)
        if ok:
            for res, sim in zip(ok, self.siglip.similarity(imgs, prompts)):
                res["raws"]["flex_siglip"] = float(sim)
        return results

    def score(self, images: List[dict], **_):
        if not self.refs:
            raise RuntimeError("prepare has not run")
        faces = [i for i in images if i.get("modality", "face") == "face"]
        flex = [i for i in images if i.get("modality") == "flex"]
        return {"results": self._score_faces(faces) + self._score_flex(flex)}
