"""Body v1 backend (metric_version 1.0.0-body-v1).

Per image: Mask R-CNN person box -> main-subject selection and
truncation check -> two independent body estimates:
  SAM 3D Body (MHR mesh: 45 shape + 28 scale components, pose, camera)
  SMPLest-X   (SMPL-X betas, pose, camera)
The reference body is the per-component median of the reference estimates in
each space (the leave-one-out target uses the same rule without the evaluated
reference). The head is excluded from every measurement.

Features, each calibrated on its own:
  body_shape_mhr    0.30  pose-neutral MHR mesh vs the reference MHR mesh
  body_shape_smplx  0.30  T-pose SMPL-X mesh vs the reference SMPL-X mesh
                          (the two are never compared with each other)
  body_proportions  0.40  SMPL-X anthropometrics (widths, lengths, girths) as
                          ratios to the neck-to-ankle height
The plan's reprojected-silhouette feature was dropped: on the reference set it
did not separate the reference body from random bodies.
Shape raw values are scale-normalized (estimated absolute height is camera
dependent), so they measure build, not size.

Code: sam-3d-body, dinov3 and SMPLest-X at pinned commits under
~/.cache/identity_eval/repos (cloned when missing). Weights come from Hugging
Face (facebook/sam-3d-body-dinov3 is gated: accept its license on the site)
and are loaded with torch.load(weights_only=True). SMPL-X model files come from
smplx_model_dir (default C:/SMPL-X). Needs a CUDA device.
"""

import contextlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import types
from typing import Dict, List, Optional

import cv2
import numpy as np

from face import hf_snapshot, list_images, read_bgr

REPO_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "identity_eval", "repos")
REPOS = {
    "sam-3d-body": ("https://github.com/facebookresearch/sam-3d-body", "b5c765a0d89d789985e186d396315e7590887b94"),
    "dinov3": ("https://github.com/facebookresearch/dinov3", "6876159a11b4df116f30f667f8c9888617df0751"),
    "SMPLest-X": ("https://github.com/SMPLCap/SMPLest-X", "fdebd887a317f9004b435c57812d1a8936295360"),
}
DEFAULT_MODELS = {
    "sam3d_body_repo": "facebook/sam-3d-body-dinov3",
    "smplestx_repo": "waanqii/SMPLest-X",
    "smplx_model_dir": "C:/SMPL-X",
}
FEATURE_WEIGHTS = {"body_shape_mhr": 0.3, "body_shape_smplx": 0.3, "body_proportions": 0.4}

PERSON_SCORE = 0.7
AMBIGUOUS_RATIO = 0.5     # a second person this large (area) makes the subject ambiguous
MIN_PERSON_PX = 256       # person box height
ANKLE_MARGIN = 1.05       # ankles may sit this far below the frame (feet cropped, legs whole)
CONTAINED = 0.6           # a detection this much inside the main box is part of the same person
HEAD_LABELS = (15, 22, 23, 24)   # SMPL-X head, jaw, eyes (smplx_parts_segm face labels)
KP = {"l_ear": 3, "r_ear": 4, "l_shoulder": 5, "r_shoulder": 6, "l_hip": 9, "r_hip": 10,
      "l_ankle": 13, "r_ankle": 14}  # SAM 3D Body mhr70 keypoints


# ----------------------------------------------------------------------
# pinned source checkouts
# ----------------------------------------------------------------------
def ensure_repo(name: str) -> str:
    url, sha = REPOS[name]
    path = os.path.join(REPO_ROOT, name)

    def head():
        try:
            return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None

    if os.path.isdir(path) and head() == sha:
        return path
    print(f"fetching {name} @ {sha[:7]}", flush=True)
    os.makedirs(path, exist_ok=True)
    git = ["git", "-C", path]
    if not os.path.isdir(os.path.join(path, ".git")):
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["remote", "add", "origin", url], check=True)
    subprocess.run(git + ["fetch", "-q", "--depth", "1", "origin", sha], check=True)
    subprocess.run(git + ["checkout", "-q", "--force", "FETCH_HEAD"], check=True)
    if head() != sha:
        raise RuntimeError(f"{name}: could not check out {sha}")
    return path


def _register_packages(repo: str, names):
    """Import the repo's top-level folders (no __init__.py) as packages ahead of
    same-named modules elsewhere on sys.path (the worker's own main.py)."""
    for n in names:
        mod = types.ModuleType(n)
        mod.__path__ = [os.path.join(repo, n)]
        sys.modules[n] = mod


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def normalized(v: np.ndarray) -> np.ndarray:
    """Center and scale a point set to unit RMS radius."""
    c = v - v.mean(axis=0)
    return c / max(float(np.sqrt((c ** 2).sum(axis=1).mean())), 1e-9)


def shape_raw(v: np.ndarray, ref: np.ndarray) -> float:
    """-mean vertex distance between scale-normalized meshes, per mille of body size."""
    return -1000.0 * float(np.linalg.norm(normalized(v) - normalized(ref), axis=1).mean())


def _inside(box, outer) -> float:
    """Fraction of box's area inside outer."""
    iw = max(0.0, min(box[2], outer[2]) - max(box[0], outer[0]))
    ih = max(0.0, min(box[3], outer[3]) - max(box[1], outer[1]))
    return iw * ih / max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)


def body_yaw(kp3d: np.ndarray) -> float:
    """Body facing angle from 3D keypoints in camera space (y down, z forward):
    0 = facing the camera, +-180 = back to the camera."""
    ls, rs, lh, rh = (kp3d[KP[k]] for k in ("l_shoulder", "r_shoulder", "l_hip", "r_hip"))
    up = (ls + rs) / 2 - (lh + rh) / 2
    across = rs - ls  # subject's left -> right
    n = np.cross(up, across)
    return math.degrees(math.atan2(n[0], -n[2]))


def body_view(yaw: float) -> str:
    a = abs(yaw)
    if a < 30:
        return "front"
    if a < 65:
        return "three_quarter"
    if a < 115:
        return "side"
    return "back"


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
class PersonDetector:
    def __init__(self):
        import torch
        import torchvision
        w = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=w).eval()
        self.torch = torch

    def to(self, device):
        self.model.to(device)

    def detect(self, rgb: np.ndarray):
        """[(box[4], score)] for people, largest first."""
        torch = self.torch
        dev = next(self.model.parameters()).device
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(dev)
        with torch.no_grad():
            out = self.model([x])[0]
        keep = (out["labels"] == 1) & (out["scores"] >= PERSON_SCORE)
        boxes = out["boxes"][keep].cpu().numpy()
        scores = out["scores"][keep].cpu().numpy()
        people = [(boxes[i], float(scores[i])) for i in range(len(boxes))]
        return sorted(people, key=lambda p: -(p[0][2] - p[0][0]) * (p[0][3] - p[0][1]))


class Sam3dBody:
    def __init__(self, repo_id: str):
        import torch
        self.torch = torch
        repo = ensure_repo("sam-3d-body")
        dino = ensure_repo("dinov3")
        os.environ["MOMENTUM_ENABLED"] = "0"   # use the TorchScript MHR shipped with the weights
        sys.path.insert(0, repo)
        from sam_3d_body.models.meta_arch import SAM3DBody
        from sam_3d_body.utils.config import get_config
        from sam_3d_body.utils.checkpoint import load_state_dict
        from sam_3d_body.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
        from sam_3d_body.data.utils.prepare_batch import prepare_batch
        from torchvision.transforms import ToTensor
        self.prepare_batch = prepare_batch
        d = hf_snapshot(repo_id)
        cfg = get_config(os.path.join(d, "model_config.yaml"))
        cfg.defrost()
        cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = os.path.join(d, "assets", "mhr_model.pt")
        cfg.freeze()
        with self._local_dinov3(dino):
            self.model = SAM3DBody(cfg)
        ck = torch.load(os.path.join(d, "model.ckpt"), map_location="cpu", weights_only=True)
        load_state_dict(self.model, ck.get("state_dict", ck), strict=False)
        # its TorchScript body model was loaded onto cuda; keep everything together
        self.model.to("cuda").eval()
        self.cfg = cfg
        self.transform = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=cfg.MODEL.IMAGE_SIZE, use_udp=False),
                                  VisionTransformWrapper(ToTensor())])
        self.faces = self.model.head_pose.faces.cpu().numpy()
        self.head_idx = self._head_vertices()

    @staticmethod
    @contextlib.contextmanager
    def _local_dinov3(path: str):
        """The backbone is built with torch.hub.load('facebookresearch/dinov3'),
        which fetches the latest code from github; build it from the pinned
        checkout instead (architecture only, weights come from the checkpoint)."""
        import torch
        orig = torch.hub.load

        def load(repo, name, *a, **k):
            if repo != "facebookresearch/dinov3":
                return orig(repo, name, *a, **k)
            sys.path.insert(0, path)
            from dinov3.hub import backbones
            k.pop("source", None)
            return getattr(backbones, name)(**k)

        torch.hub.load = load
        try:
            yield
        finally:
            torch.hub.load = orig

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, device):
        self.model.to(device)

    def _mhr(self, shape, scale, global_rot=None, body_pose=None, hand=None, keypoints=False):
        """MHR mesh in its own frame (y up, meters). Zero pose when pose is None."""
        torch = self.torch
        head = self.model.head_pose
        dev = self.device
        t = lambda a, n: (torch.as_tensor(a, dtype=torch.float32, device=dev).reshape(1, -1)
                          if a is not None else torch.zeros(1, n, device=dev))
        with torch.no_grad():
            out = head.mhr_forward(global_trans=torch.zeros(1, 3, device=dev), global_rot=t(global_rot, 3),
                                   body_pose_params=t(body_pose, 133), hand_pose_params=t(hand, 108),
                                   scale_params=t(scale, 28), shape_params=t(shape, 45),
                                   expr_params=torch.zeros(1, 72, device=dev), return_keypoints=keypoints)
        if keypoints:
            return out[0][0].cpu().numpy(), out[1][0, :70].cpu().numpy()
        return out[0].cpu().numpy()

    def _head_vertices(self) -> np.ndarray:
        """Vertices above the chin line of the zero-shape MHR body (topology fixed)."""
        v, kp = self._mhr(np.zeros(45), np.zeros(28), keypoints=True)
        shoulder = (kp[KP["l_shoulder"], 1] + kp[KP["r_shoulder"], 1]) / 2
        ear = (kp[KP["l_ear"], 1] + kp[KP["r_ear"], 1]) / 2
        return np.where(v[:, 1] > shoulder + 0.45 * (ear - shoulder))[0]

    def neutral(self, shape, scale) -> np.ndarray:
        return self._mhr(shape, scale)

    def estimate(self, rgb: np.ndarray, box: np.ndarray) -> dict:
        torch = self.torch
        from sam_3d_body.utils import recursive_to
        batch = self.prepare_batch(rgb, self.transform, box.reshape(1, 4).astype(np.float32))
        batch = recursive_to(batch, self.device)
        self.model._initialize_batch(batch)
        with torch.no_grad():
            out = self.model.run_inference(rgb, batch, inference_type="body")["mhr"]
        g = lambda k: out[k][0].float().cpu().numpy()
        return {"vertices": g("pred_vertices"), "cam_t": g("pred_cam_t"), "focal": float(g("focal_length")),
                "kp3d": g("pred_keypoints_3d"), "kp2d": g("pred_keypoints_2d"), "shape": g("shape"),
                "scale": g("scale"), "global_rot": g("global_rot"), "body_pose": g("body_pose"), "hand": g("hand")}


class SmplestX:
    PACKAGES = ("main", "models", "utils", "human_models")

    def __init__(self, repo_id: str, smplx_dir: str):
        import torch
        self.torch = torch
        repo = ensure_repo("SMPLest-X")
        hm = self._human_models(smplx_dir)
        _register_packages(repo, self.PACKAGES)
        sys.path.insert(0, repo)
        from main.config import Config
        from human_models.human_models import SMPLX
        from utils.data_utils import process_bbox, generate_patch_image
        self.process_bbox, self.generate_patch_image = process_bbox, generate_patch_image
        d = hf_snapshot(repo_id, ["config_base.py", "smplest_x_h.pth.tar"])
        cfg = Config.load_config(os.path.join(d, "config_base.py"))
        cfg.update_config({"model": {"human_model_path": hm, "encoder_pretrained_model_path": ""}})
        SMPLX(hm)
        from models.SMPLest_X import get_model
        self.model = get_model(cfg, "test")
        ck = torch.load(os.path.join(d, "smplest_x_h.pth.tar"), map_location="cpu", weights_only=True)
        sd = {k.replace("module.", "").replace("backbone", "encoder").replace("body_rotation_net", "body_regressor")
              .replace("hand_rotation_net", "hand_regressor"): v for k, v in ck["network"].items()}
        missing = [k for k in self.model.load_state_dict(sd, strict=False).missing_keys
                   if not k.startswith("smplx_layer.")]
        if missing:
            raise RuntimeError(f"SMPLest-X checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
        self.model.eval()
        self.cfg = cfg

    @staticmethod
    def _human_models(smplx_dir: str) -> str:
        """Model folder in the layout the repo expects: the SMPL-X npz files,
        the MANO/FLAME correspondences, and a zero J14 regressor (only used by
        the repo's benchmark evaluation)."""
        hm = os.path.join(os.path.expanduser("~"), ".cache", "identity_eval", "smplestx_human_models")
        os.makedirs(os.path.join(hm, "smplx"), exist_ok=True)
        for g in ("NEUTRAL", "MALE", "FEMALE"):
            dst = os.path.join(hm, "smplx", f"SMPLX_{g}.npz")
            if not os.path.exists(dst):
                src = os.path.join(smplx_dir, "smplx", f"SMPLX_{g}.npz")
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy(src, dst)
        for f in ("MANO_SMPLX_vertex_ids.pkl", "SMPL-X__FLAME_vertex_ids.npy"):
            dst = os.path.join(hm, "smplx", f)
            if not os.path.exists(dst):
                shutil.copy(os.path.join(smplx_dir, f), dst)
        j14 = os.path.join(hm, "smplx", "SMPLX_to_J14.pkl")
        if not os.path.exists(j14):
            with open(j14, "wb") as fh:
                pickle.dump(np.zeros((14, 10475), dtype=np.float32), fh)
        return hm

    def to(self, device):
        self.model.to(device)

    def estimate(self, rgb: np.ndarray, box: np.ndarray) -> dict:
        torch = self.torch
        h, w = rgb.shape[:2]
        xywh = np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]], dtype=np.float32)
        bbox = self.process_bbox(xywh, w, h, self.cfg.model.input_img_shape, getattr(self.cfg.data, "bbox_ratio", 1.25))
        patch, _, _ = self.generate_patch_image(rgb.astype(np.float32), bbox, 1.0, 0.0, False,
                                                self.cfg.model.input_img_shape)
        dev = next(self.model.parameters()).device
        x = torch.from_numpy(patch).permute(2, 0, 1)[None].float().to(dev) / 255
        with torch.no_grad():
            out = self.model({"img": x}, {}, {}, "test")
        return {"betas": out["smplx_shape"][0].float().cpu().numpy()}


def _numpy_unpickle(path: str):
    """Pickle loader that only builds numpy arrays (SMPL-X part labels)."""
    class Loader(pickle.Unpickler):
        ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
                   ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode")}

        def find_class(self, module, name):
            if (module, name) not in self.ALLOWED:
                raise pickle.UnpicklingError(f"{module}.{name} is not allowed")
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return Loader(f, encoding="latin1").load()


class SmplxBody:
    """SMPL-X neutral T-pose body from betas, plus anthropometrics."""

    def __init__(self, smplx_dir: str):
        import torch
        import smplx
        self.torch = torch
        self.layer = smplx.create(smplx_dir, "smplx", gender="neutral", use_pca=False, num_betas=10).eval()
        self.faces = self.layer.faces.astype(np.int64)
        segm = _numpy_unpickle(os.path.join(smplx_dir, "smplx_parts_segm.pkl"))["segm"]
        vlabel = np.full(self.layer.get_num_verts(), -1)
        vlabel[self.faces.reshape(-1)] = np.repeat(segm, 3)
        self.vlabel = vlabel
        self.body_idx = np.where(~np.isin(vlabel, HEAD_LABELS))[0]

    def mesh(self, betas):
        torch = self.torch
        with torch.no_grad():
            out = self.layer(betas=torch.as_tensor(betas, dtype=torch.float32).reshape(1, -1))
        return out.vertices[0].numpy(), out.joints[0, :55].numpy()

    def _girth(self, v, height, labels) -> float:
        sel = np.isin(self.vlabel, labels) & (np.abs(v[:, 1] - height) < 0.012)
        pts = v[sel][:, [0, 2]]
        if len(pts) < 3:
            return float("nan")
        hull = cv2.convexHull((pts * 1e4).astype(np.float32))
        return cv2.arcLength(hull, True) / 1e4

    def proportions(self, betas) -> np.ndarray:
        v, j = self.mesh(betas)
        d = lambda a, b: float(np.linalg.norm(j[a] - j[b]))
        stature = j[12, 1] - (j[7, 1] + j[8, 1]) / 2   # neck to ankles
        torso = (0, 3, 6, 9)
        m = [
            d(16, 17),                                    # shoulder width
            d(0, 12),                                     # torso length
            (d(1, 4) + d(2, 5)) / 2,                      # thigh length
            (d(4, 7) + d(5, 8)) / 2,                      # shin length
            (d(16, 18) + d(17, 19)) / 2,                  # upper arm
            (d(18, 20) + d(19, 21)) / 2,                  # forearm
            self._girth(v, j[9, 1], torso),               # chest
            self._girth(v, j[3, 1], torso),               # waist
            self._girth(v, j[0, 1] - 0.04, torso + (1, 2)),   # hips
            self._girth(v, (j[1, 1] + j[4, 1]) / 2, (1,)),    # thigh
        ]
        return np.array(m) / stature


def proportions_raw(p: np.ndarray, ref: np.ndarray) -> float:
    """-mean absolute log ratio of the measurements, in percent."""
    ok = np.isfinite(p) & np.isfinite(ref) & (p > 0) & (ref > 0)
    return -100.0 * float(np.abs(np.log(p[ok] / ref[ok])).mean()) if ok.any() else float("nan")


# ----------------------------------------------------------------------
# backend
# ----------------------------------------------------------------------
class BodyBackend:
    name = "body-v1"
    metric_version = "1.0.0-body-v1"
    features = list(FEATURE_WEIGHTS)
    weights = dict(FEATURE_WEIGHTS)

    def __init__(self, role: str, device: str, models: Optional[dict] = None):
        import torch
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("body scoring needs a CUDA GPU (SMPLest-X runs on cuda only); set body_gpu")
        self.models = dict(DEFAULT_MODELS, **(models or {}))
        smplx_dir = self.models["smplx_model_dir"] or DEFAULT_MODELS["smplx_model_dir"]
        if not os.path.exists(os.path.join(smplx_dir, "smplx", "SMPLX_NEUTRAL.npz")):
            raise RuntimeError(f"SMPL-X model not found: {smplx_dir}/smplx/SMPLX_NEUTRAL.npz "
                               "(extract SMPLX_NEUTRAL.npz from models_smplx_v1_1.zip there)")
        self.torch = torch
        # SAM 3D Body loads its TorchScript body model onto cuda while building
        self.detector = PersonDetector()
        self.sam = Sam3dBody(self.models["sam3d_body_repo"])
        self.smplestx = SmplestX(self.models["smplestx_repo"], smplx_dir)
        self.smplx = SmplxBody(smplx_dir)
        self.model_ids = {"detector": "torchvision maskrcnn_resnet50_fpn_v2 COCO_V1",
                          "sam3d_body": self.models["sam3d_body_repo"], "smplestx": self.models["smplestx_repo"],
                          "repos": {k: v[1][:7] for k, v in REPOS.items()}}
        self.refs: List[dict] = []
        self._on_gpu = None
        self.to_cpu()

    # -- device handling --------------------------------------------------
    def to_gpu(self, **_):
        if self._on_gpu is True:
            return {}
        for m in (self.detector, self.sam, self.smplestx):
            m.to("cuda")
        self._on_gpu = True
        return {}

    def to_cpu(self, **_):
        if self._on_gpu is False:
            return {}
        for m in (self.detector, self.sam, self.smplestx):
            m.to("cpu")
        self.torch.cuda.empty_cache()
        self._on_gpu = False
        return {}

    # -- per image --------------------------------------------------------
    def _analyze(self, path: str) -> dict:
        bgr = read_bgr(path)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        people = self.detector.detect(rgb)
        if people:
            main = people[0][0]
            people = people[:1] + [p for p in people[1:] if _inside(p[0], main) < CONTAINED]
        if not people:
            return {"status": "NO_BODY", "reason": "no person detected"}
        areas = [(p[0][2] - p[0][0]) * (p[0][3] - p[0][1]) for p in people]
        if len(people) > 1 and areas[1] >= AMBIGUOUS_RATIO * areas[0]:
            return {"status": "NO_BODY", "reason": f"ambiguous: {len(people)} people of similar size"}
        box, score = people[0]
        if box[3] - box[1] < MIN_PERSON_PX:
            return {"status": "NO_BODY", "reason": f"person too small ({int(box[3] - box[1])}px tall)"}
        sam = self.sam.estimate(rgb, box)
        ankles = sam["kp2d"][[KP["l_ankle"], KP["r_ankle"]]]
        if (ankles[:, 1] > h * ANKLE_MARGIN).any():
            return {"status": "BODY_CROPPED", "reason": "feet out of frame"}
        smx = self.smplestx.estimate(rgb, box)
        yaw = body_yaw(sam["kp3d"])
        return {"status": "VALID", "sam": sam, "betas": smx["betas"],
                "props": self.smplx.proportions(smx["betas"]),
                "extra": {"yaw": round(yaw, 1), "view": body_view(yaw), "det_score": round(score, 3),
                          "person_px": int(box[3] - box[1])}}

    # -- reference body -----------------------------------------------------
    def _canonical(self, exclude: Optional[int] = None) -> dict:
        rows = [r for i, r in enumerate(self.refs) if i != exclude]
        shape = np.median([r["sam"]["shape"] for r in rows], axis=0)
        scale = np.median([r["sam"]["scale"] for r in rows], axis=0)
        betas = np.median([r["betas"] for r in rows], axis=0)
        mhr = self.sam.neutral(shape, scale)
        sx, _ = self.smplx.mesh(betas)
        return {"shape": shape, "scale": scale, "mhr": mhr, "smplx": sx, "props": self.smplx.proportions(betas)}

    def _raws(self, a: dict, canon: dict) -> Dict[str, float]:
        body = np.setdiff1d(np.arange(len(canon["mhr"])), self.sam.head_idx)
        own_mhr = self.sam.neutral(a["sam"]["shape"], a["sam"]["scale"])
        own_sx, _ = self.smplx.mesh(a["betas"])
        return {
            "body_shape_mhr": shape_raw(own_mhr[body], canon["mhr"][body]),
            "body_shape_smplx": shape_raw(own_sx[self.smplx.body_idx], canon["smplx"][self.smplx.body_idx]),
            "body_proportions": proportions_raw(a["props"], canon["props"]),
        }

    def prepare(self, references: List[str], cache_dir: str, **_):
        paths = []
        for r in references:
            paths.extend(list_images(r) if os.path.isdir(r) else [r])
        excluded, refs = [], []
        for p in paths:
            try:
                a = self._analyze(p)
            except Exception as e:
                a = {"status": "FAILED", "reason": f"{type(e).__name__}: {e}"}
            if a["status"] != "VALID":
                excluded.append({"path": p, "reason": a["reason"]})
            else:
                a["path"] = p
                refs.append(a)
        if len(refs) < 3:
            raise RuntimeError(f"need at least 3 reference images with a usable full body, found {len(refs)} "
                               f"({len(excluded)} excluded)")
        self.refs = refs
        self.canon = self._canonical()
        loo = {f: [] for f in self.features}
        for i, ref in enumerate(refs):
            raws = self._raws(ref, self._canonical(exclude=i))
            for f in self.features:
                loo[f].append(raws[f])
        views = {}
        for r in refs:
            views[r["extra"]["view"]] = views.get(r["extra"]["view"], 0) + 1
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(os.path.join(cache_dir, "body_reference.npz"), mhr_shape=self.canon["shape"],
                 mhr_scale=self.canon["scale"], smplx_betas=np.median([r["betas"] for r in refs], axis=0),
                 proportions=self.canon["props"])
        with open(os.path.join(cache_dir, "body_refs.json"), "w", encoding="utf-8") as f:
            json.dump({"references": [{"path": r["path"], **r["extra"]} for r in refs], "excluded": excluded,
                       "views": views, "models": self.model_ids,
                       "loo": {k: [round(x, 4) for x in v] for k, v in loo.items()}}, f, indent=2)
        print(f"body references: {len(refs)} usable {views}, {len(excluded)} excluded", flush=True)
        return {
            "metric_version": self.metric_version,
            "reference_count": len(refs),
            "excluded": excluded,
            "views": views,
            "features": {f: {"loo": loo[f]} for f in self.features},
        }

    def score(self, images: List[dict], **_):
        if not self.refs:
            raise RuntimeError("prepare has not run")
        results = []
        for item in images:
            try:
                a = self._analyze(item["path"])
            except Exception as e:
                a = {"status": "FAILED", "reason": f"{type(e).__name__}: {e}"}
            res = {"slot": item["slot"], "status": a["status"], "reason": a.get("reason"), "raws": {}}
            if a["status"] == "VALID":
                res["extra"] = a["extra"]
                res["raws"] = {k: v for k, v in self._raws(a, self.canon).items() if np.isfinite(v)}
            results.append(res)
        return {"results": results}
