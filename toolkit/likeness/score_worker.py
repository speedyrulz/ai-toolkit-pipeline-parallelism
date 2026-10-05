"""Character likeness scoring worker.

Runs inside a ComfyUI installation's own Python (it imports ComfyUI's SAM3D
Body implementation and the character-similarity custom nodes), NOT inside
ai-toolkit's environment. The trainer starts it once per run with
toolkit/likeness/scorer.py and feeds it one JSON job per line on stdin:

    {"step": 500, "folder": "/path/to/this/rounds/samples"}

For each job it scores every image in the folder against the reference
character with the same nodes the character_similarity workflow uses
(Score Face Similarity, Score SAM3D Body Shape Similarity, Score Body Detail
Similarity, Character Score Summary), writes a per-step report and a CSV row
into --out, and prints one result line on stdout:

    LIKENESS_RESULT {"step": 500, "face": 61.2, ...}

Everything else it prints goes to stderr. It exits when stdin closes.

Required custom nodes in <comfyui>/custom_nodes: face_similarity_score.py,
sam3d_body_shape_score.py, body_detail_score.py, character_score_tools.py and
the comfyui_faceanalysis pack.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
import traceback

RESULT_PREFIX = "LIKENESS_RESULT "
READY_PREFIX = "LIKENESS_READY"
NOT_SCORED = -1.0

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

CSV_FIELDS = [
    "step", "images", "overall", "face", "body_shape", "body_detail",
    "proportion", "build", "face_scored", "body_scored", "detail_scored",
    "seconds",
]


def _emit(line):
    sys.__stdout__.write(line + "\n")
    sys.__stdout__.flush()


def _log(msg):
    sys.stderr.write(f"[likeness] {msg}\n")
    sys.stderr.flush()


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--comfyui", required=True, help="ComfyUI root folder")
    p.add_argument("--refs", required=True, help="folder of reference images")
    p.add_argument("--out", required=True, help="folder for reports and the CSV")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--cpu-threads", type=int, default=0)
    # gpu to score on, numbered like nvidia-smi (PCI bus order); -1 = default
    p.add_argument("--gpu-index", type=int, default=-1)
    # auto: the bf16 checkpoint on cpu, int8 on gpu. The int8 (convrot)
    # file has no fast cpu path: measured 304s per body fit on cpu vs 7-10s
    # for bf16 on the same 6 threads
    p.add_argument("--sam3d-model", default="auto")
    p.add_argument("--clip-vision-model", default="dinov2_large.safetensors")
    p.add_argument("--face-library", default="insightface")
    p.add_argument("--proportion-tolerance", type=float, default=3.0)
    p.add_argument("--build-tolerance", type=float, default=8.0)
    p.add_argument("--proportion-weight", type=float, default=0.5)
    p.add_argument("--weights", default="1,1,1",
                   help="face,body_shape,body_detail weights for the overall score")
    return p.parse_args()


def _bootstrap_comfy(root, device):
    """Import ComfyUI as a library: no server, no prompt queue."""
    root = os.path.abspath(root)
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    # ComfyUI only parses argv when its main.py enables it; set the options
    # we need on the defaults before model_management reads them at import
    import comfy.options  # noqa: F401
    from comfy.cli_args import args
    if device == "cpu":
        args.cpu = True
    import comfy.model_management  # noqa: F401
    import folder_paths  # noqa: F401


def _load_module(name, path, package_dir=None):
    if package_dir is not None:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(package_dir, "__init__.py"),
            submodule_search_locations=[package_dir],
        )
    else:
        spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise FileNotFoundError(path or package_dir)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_model(folder_type, name):
    import folder_paths
    path = folder_paths.get_full_path(folder_type, name)
    if path is not None:
        return path
    available = folder_paths.get_filename_list(folder_type)
    # tolerate the common rename variants (dinov2_large vs "dinov2-large ")
    want = "".join(ch for ch in name.lower() if ch.isalnum())
    for candidate in available:
        if "".join(ch for ch in candidate.lower() if ch.isalnum()) == want:
            return folder_paths.get_full_path(folder_type, candidate)
    raise FileNotFoundError(
        f"{name!r} not found in models/{folder_type}; available: {available}"
    )


def _list_images(folder):
    import re

    def natural(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")
    ]
    return sorted(files, key=natural)


def _out(result):
    """NodeOutput -> tuple of outputs."""
    return result.result if hasattr(result, "result") else result


def _mean(values):
    vals = [float(v) for v in values if v is not None and float(v) != NOT_SCORED]
    return (sum(vals) / len(vals)) if vals else None, len(vals)


class Scorer:
    def __init__(self, a):
        self.a = a
        import torch
        if a.cpu_threads > 0:
            torch.set_num_threads(a.cpu_threads)

        root = os.path.abspath(a.comfyui)
        cn = os.path.join(root, "custom_nodes")
        from comfy_extras.nodes_sam3d_body import SAM3DBody_Loader, SAM3DBody_Predict
        from comfy_extras.nodes_dataset import load_and_process_images
        import comfy.clip_vision

        self.load_images = load_and_process_images
        self.Predict = SAM3DBody_Predict
        face_mod = _load_module("aitk_face_similarity_score", os.path.join(cn, "face_similarity_score.py"))
        shape_mod = _load_module("aitk_sam3d_body_shape_score", os.path.join(cn, "sam3d_body_shape_score.py"))
        detail_mod = _load_module("aitk_body_detail_score", os.path.join(cn, "body_detail_score.py"))
        tools_mod = _load_module("aitk_character_score_tools", os.path.join(cn, "character_score_tools.py"))
        self.FaceScore = face_mod.FaceSimilarityScore
        self.ShapeScore = shape_mod.SAM3DBodyShapeScore
        self.DetailScore = detail_mod.BodyDetailScore
        self.Summary = tools_mod.CharacterScoreSummary

        t0 = time.time()
        _log("loading SAM3D Body model")
        sam3d_name = os.path.basename(_resolve_model("detection", self._pick_sam3d(a)))
        _log(f"SAM3D Body checkpoint: {sam3d_name}")
        self.sam3d = _out(SAM3DBody_Loader.execute(sam3d_name))[0]

        _log("loading image encoder")
        self.clip_vision = comfy.clip_vision.load(_resolve_model("clip_vision", a.clip_vision_model))

        _log("loading face models")
        fa_dir = None
        for entry in os.listdir(cn):
            if entry.lower() == "comfyui_faceanalysis":
                fa_dir = os.path.join(cn, entry)
        if fa_dir is None:
            raise FileNotFoundError("comfyui_faceanalysis custom node pack not found")
        fa_pkg = _load_module("aitk_comfyui_faceanalysis", None, package_dir=fa_dir)
        fa_cls = getattr(fa_pkg, "NODE_CLASS_MAPPINGS", {}).get("FaceAnalysisModels")
        if fa_cls is None:
            fa_mod = sys.modules.get("aitk_comfyui_faceanalysis.faceanalysis")
            fa_cls = fa_mod.FaceAnalysisModels
        provider = "CPU" if a.device == "cpu" else "CUDA"
        self.face_models = fa_cls().load_models(a.face_library, provider)[0]

        _log(f"loading references from {a.refs}")
        ref_files = _list_images(a.refs)
        if len(ref_files) == 0:
            raise ValueError(f"no reference images in {a.refs}")
        self.ref_images = self.load_images(ref_files, a.refs)
        t1 = time.time()
        self.ref_poses = [self._predict(img) for img in self.ref_images]
        _log(f"reference body fits: {time.time() - t1:.0f}s for {len(ref_files)} images")
        if a.device == "gpu":
            import torch
            _log(f"scoring on gpu: {torch.cuda.get_device_name(0)}")
        _log(f"ready: {len(ref_files)} references, setup took {time.time() - t0:.0f}s")

        w = [float(x) for x in a.weights.split(",")]
        self.weights = (w + [1.0, 1.0, 1.0])[:3]

    @staticmethod
    def _pick_sam3d(a):
        if a.sam3d_model != "auto":
            if a.device == "cpu" and "int8" in a.sam3d_model.lower():
                _log("warning: int8 SAM3D checkpoints run ~40x slower on cpu than bf16")
            return a.sam3d_model
        import folder_paths
        files = [f for f in folder_paths.get_filename_list("detection") if "sam_3d_body" in f.lower()]
        if not files:
            raise FileNotFoundError("no sam_3d_body checkpoint in models/detection")
        prefer = ("bf16", "fp16", "fp32") if a.device == "cpu" else ("int8", "bf16", "fp16")
        for tag in prefer:
            for f in files:
                if tag in f.lower():
                    return f
        return files[0]

    def _predict(self, image):
        # hand refinement off, fov 0, batch 64: the workflow's settings
        return _out(self.Predict.execute(self.sam3d, image, run_hand_refinement=False,
                                         fov=0.0, batch_size=64))[0]

    def score(self, step, folder):
        a = self.a
        t0 = time.time()
        files = _list_images(folder)
        if not files:
            raise ValueError(f"no images to score in {folder}")
        images = self.load_images(files, folder)
        poses = [self._predict(img) for img in images]
        t_fit = time.time()
        _log(f"step {step}: body fits {t_fit - t0:.0f}s for {len(files)} images")

        face = _out(self.FaceScore.execute(
            [self.face_models], self.ref_images, images, ["score_each"], [0.0], [0.0]))
        shape = _out(self.ShapeScore.execute(
            [self.sam3d], self.ref_poses, poses, ["score_each"],
            [a.proportion_tolerance], [a.build_tolerance], [a.proportion_weight], [0], [0]))
        t_face = time.time()
        detail = _out(self.DetailScore.execute(
            [self.clip_vision], [self.sam3d], self.ref_images, self.ref_poses,
            images, poses, ["score_each"], [0.0], [0.0], [0], [0]))
        _log(f"step {step}: face+shape {t_face - t_fit:.0f}s, detail {time.time() - t_face:.0f}s")

        face_scores = list(face[0])
        proportion_scores, build_scores, shape_scores = list(shape[0]), list(shape[1]), list(shape[2])
        detail_scores = list(detail[0])

        summary = _out(self.Summary.execute(
            face_score=face_scores, body_shape_score=shape_scores,
            body_detail_score=detail_scores, image_names=files,
            face_average_report=[face[4]], body_shape_average_report=[shape[-1]],
            body_detail_average_report=[detail[4]],
        ))

        face_avg, n_face = _mean(face_scores)
        shape_avg, n_body = _mean(shape_scores)
        detail_avg, n_detail = _mean(detail_scores)
        prop_avg, _ = _mean(proportion_scores)
        build_avg, _ = _mean(build_scores)

        parts = [(v, w) for v, w in zip((face_avg, shape_avg, detail_avg), self.weights)
                 if v is not None and w > 0]
        overall = (sum(v * w for v, w in parts) / sum(w for _, w in parts)) if parts else None
        seconds = time.time() - t0

        def r(v):
            return None if v is None else round(float(v), 2)

        result = {
            "step": int(step), "images": len(files), "overall": r(overall),
            "face": r(face_avg), "body_shape": r(shape_avg), "body_detail": r(detail_avg),
            "proportion": r(prop_avg), "build": r(build_avg),
            "face_scored": n_face, "body_scored": n_body, "detail_scored": n_detail,
            "seconds": round(seconds, 1),
        }

        os.makedirs(a.out, exist_ok=True)
        report_path = os.path.join(a.out, f"step_{int(step):09d}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Character likeness, step {step}\n")
            f.write(f"Overall likeness: {'-' if overall is None else format(overall, '.1f')}"
                    f"  (weights face/body shape/body detail = "
                    f"{'/'.join(format(x, 'g') for x in self.weights)})\n")
            f.write(f"Samples folder: {folder}\n\n")
            f.write(str(summary[0]) + "\n\n")
            # the detailed report repeats the summary table; keep only the
            # averaged breakdown sections after it
            detailed = str(summary[1])
            rule_at = detailed.find("-" * 78)
            if rule_at >= 0:
                f.write(detailed[rule_at:] + "\n\n")
            for title, rep in (("FACE, PER IMAGE", face[3]), ("BODY SHAPE, PER IMAGE", shape[-2]),
                               ("BODY DETAIL, PER IMAGE", detail[3])):
                f.write("-" * 78 + f"\n{title}\n" + "-" * 78 + f"\n{rep}\n\n")

        csv_path = os.path.join(a.out, "likeness_scores.csv")
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({k: ("" if result[k] is None else result[k]) for k in CSV_FIELDS})

        result["report"] = report_path
        return result


def main():
    a = _parse_args()
    if a.device == "gpu" and a.gpu_index >= 0:
        # must be set before torch initializes CUDA; PCI order matches
        # nvidia-smi and the trainer UI's gpu numbering
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu_index)
    _bootstrap_comfy(a.comfyui, a.device)
    import torch
    # ComfyUI's executor runs every node under inference mode; the nodes
    # assume it (they call .numpy() on model outputs)
    with torch.inference_mode():
        return _serve(a)


def _serve(a):
    try:
        scorer = Scorer(a)
    except Exception as e:
        traceback.print_exc()
        _emit(RESULT_PREFIX + json.dumps({"fatal": f"{type(e).__name__}: {e}"}))
        return 1
    _emit(READY_PREFIX)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError:
            _log(f"ignoring bad job line: {line!r}")
            continue
        if job.get("cmd") == "quit":
            break
        step = job.get("step")
        try:
            result = scorer.score(step, job["folder"])
        except Exception as e:
            traceback.print_exc()
            result = {"step": step, "error": f"{type(e).__name__}: {e}"}
        _emit(RESULT_PREFIX + json.dumps(result))
    return 0


if __name__ == "__main__":
    # keep stdout clean for result lines: anything the nodes print goes to stderr
    sys.stdout = sys.stderr
    sys.exit(main())
