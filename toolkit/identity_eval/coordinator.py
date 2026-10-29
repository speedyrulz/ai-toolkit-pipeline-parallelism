"""Trainer-side identity eval coordinator.

Owns the eval suites, the scoring workers, calibration (reference targets and
the step-0 paired baseline) and per-round results. The trainer supplies two
hooks: render(items) to generate images, and gpu_session() to free the cards
(unload the training model and the sample helper) while gpu workers score.

Layout under <save_root>/identity_eval/ (docs/identity_eval_plan.md):
  manifest.json, anchors.json, results.jsonl, worker_*.log
  calibration/<feature caches>
  baseline/<modality>/<slot>.png
  step_XXXXXXXXX/<modality>/<slot>.png, DONE, per_image.json, checkpoint_summary.json
"""

import contextlib
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

from toolkit.print import print_acc
from toolkit.identity_eval import manifest as M
from toolkit.identity_eval import scoring as S
from toolkit.identity_eval.proc import ChildError, JsonProcess
from toolkit.identity_eval.suites import build_slots, slots_signature

ROLES = ("face", "body")
WORKER_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker", "main.py")


def gpu_env(gpu: Optional[int]) -> Dict[str, str]:
    """Child env pinned to one physical card (PCI order, like the UI), or no
    card at all for cpu."""
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None or gpu < 0 else str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    return env


class IdentityEval:
    def __init__(self, cfg, sample_config, save_root: str, trainer_gpu: int, second_gpu: Optional[int],
                 use_helper: bool, default_references: str, model_fingerprint: dict,
                 prompt_fn: Callable[[str], str]):
        self.cfg = cfg
        self.sample_config = sample_config
        self.root = os.path.join(save_root, "identity_eval")
        self.prompt_fn = prompt_fn
        self.slots = build_slots(cfg, sample_config, use_helper=use_helper)
        self.uses_helper_renderer = use_helper
        self.slot_by_id = {s["id"]: s for s in self.slots}
        self.scored_slots = [s for s in self.slots if s["modality"] in ROLES]
        self.model_fingerprint = model_fingerprint

        face_gpu = cfg.face_gpu if cfg.face_gpu is not None else trainer_gpu
        body_gpu = cfg.body_gpu if cfg.body_gpu is not None else (
            second_gpu if second_gpu is not None else trainer_gpu)
        self.gpus = {"face": face_gpu, "body": body_gpu}
        self.references = {
            "face": cfg.face_references or default_references,
            "body": cfg.body_references or default_references,
        }
        self.workers: Dict[str, JsonProcess] = {}
        self.features: Dict[str, List[str]] = {}
        self.metric_versions: Dict[str, str] = {}
        self.anchors: Optional[dict] = None
        self.calibration_id: Optional[str] = None
        self.active = False
        self._pending: List[dict] = []   # async (cpu) rounds in flight
        self._ready_metrics: List[dict] = []

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    @property
    def uses_gpu(self) -> bool:
        return any(g is not None and g >= 0 for g in self.gpus.values())

    def _start_workers(self):
        python = self.cfg.evaluator_python or sys.executable
        for role in ROLES:
            gpu = self.gpus[role]
            device = "cuda" if gpu is not None and gpu >= 0 else "cpu"
            where = f"gpu {gpu}" if device == "cuda" else "cpu"
            print_acc(f"Identity eval: starting {role} worker on {where}")
            w = JsonProcess(
                f"identity {role} worker",
                [python, "-u", WORKER_MAIN, "--role", role, "--device", device],
                os.path.join(self.root, f"worker_{role}.log"),
                env=gpu_env(gpu),
            )
            w.start()
            info = w.request("info", timeout=60)
            self.workers[role] = w
            self.features[role] = info["features"]
            self.metric_versions[role] = info["metric_version"]

    def _fingerprint(self) -> dict:
        return {
            "model": self.model_fingerprint,
            "slots": slots_signature(self.slots),
            "sampler": {
                "sampler": self.sample_config.sampler,
                "guidance_rescale": self.sample_config.guidance_rescale,
                "network_multiplier": self.sample_config.network_multiplier,
            },
            "references": {r: M.hash_images(M.list_images([self.references[r]])) for r in ROLES},
            "metric_versions": self.metric_versions,
            "target_quantile": self.cfg.target_quantile,
        }

    def start(self, render: Callable[[List[dict]], None], gpu_session) -> bool:
        """Start workers and load or build the calibration. Returns whether
        identity eval is active for this run."""
        os.makedirs(self.root, exist_ok=True)
        for role in ROLES:
            if not M.list_images([self.references[role]]):
                print_acc(f"Identity eval disabled: no {role} reference images in {self.references[role]!r}")
                return False
        try:
            self._start_workers()
        except ChildError as e:
            print_acc(f"Identity eval disabled, a scoring worker could not start: {e}")
            self.close()
            return False

        fingerprint = self._fingerprint()
        cal_id = M.calibration_id(fingerprint)
        saved = M.read_json(os.path.join(self.root, "manifest.json"))
        anchors = M.read_json(os.path.join(self.root, "anchors.json"))
        if saved is not None and anchors is not None:
            if saved.get("calibration_id") != cal_id:
                print_acc(
                    "Identity eval disabled: CALIBRATION_INVALID. The model, suites, sampler, references or "
                    "scoring backend changed since this run was calibrated; scores would not be comparable. "
                    f"Delete {self.root} to recalibrate as a new metric version."
                )
                self.close()
                return False
            self.anchors = anchors
            self.calibration_id = cal_id
            self._prepare_workers(gpu_session)  # templates are rebuilt; anchors are reused
            print_acc(f"Identity eval: reusing calibration {cal_id[:19]}")
        else:
            if not self._calibrate(render, gpu_session, fingerprint, cal_id):
                self.close()
                return False
        self.active = True
        return True

    def _prepare_workers(self, gpu_session) -> Dict[str, dict]:
        cal_dir = os.path.join(self.root, "calibration")
        with self._session(gpu_session):
            ids = {r: self.workers[r].send("prepare", references=[self.references[r]],
                                            cache_dir=os.path.join(cal_dir, r)) for r in ROLES}
            return {r: self.workers[r].wait(ids[r]) for r in ROLES}

    def _calibrate(self, render, gpu_session, fingerprint: dict, cal_id: str) -> bool:
        print_acc(f"Identity eval: rendering the step-0 baseline ({len(self.slots)} images, network off)")
        items = self._items(os.path.join(self.root, "baseline"), multiplier=0.0)
        render(items)
        with self._session(gpu_session):
            try:
                prepared = self._prepare_workers(None)
                raw = self._score_images(self._image_list(os.path.join(self.root, "baseline")))
            except ChildError as e:
                print_acc(f"Identity eval disabled, calibration failed: {e}")
                return False

        targets, loo = {}, {}
        for role in ROLES:
            for feat, data in prepared[role]["features"].items():
                loo[feat] = data["loo"]
                targets[feat] = S.target(data["loo"], self.cfg.target_quantile)
        baselines = {sid: r["raws"] for sid, r in raw.items() if r["status"] == S.VALID}
        baseline_status = {sid: {"status": r["status"], "reason": r.get("reason")} for sid, r in raw.items()}
        self.anchors = {
            "targets": targets,
            "loo": loo,
            "baselines": baselines,
            "baseline_status": baseline_status,
            "reference_counts": {r: prepared[r]["reference_count"] for r in ROLES},
            "excluded_references": {r: prepared[r]["excluded"] for r in ROLES},
        }
        self.calibration_id = cal_id
        M.write_json(os.path.join(self.root, "anchors.json"), self.anchors)
        M.write_json(os.path.join(self.root, "manifest.json"), {
            "calibration_id": cal_id,
            "metric_versions": self.metric_versions,
            "features": self.features,
            "gpus": self.gpus,
            "references": self.references,
            "holdout_references": self.cfg.holdout_references,
            "fingerprint": fingerprint,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        for feat, t in targets.items():
            print_acc(f"Identity eval: {feat} target {t if t is None else round(t, 4)} "
                      f"from {len(loo[feat])} reference comparisons")
        uncal = [f for f, t in targets.items() if t is None]
        if uncal:
            print_acc(f"Identity eval: features without a usable target (too few references): {uncal}")
        if not self.cfg.holdout_references:
            print_acc("Identity eval: no holdout references; the 100 target comes from the training images")
        return True

    # ------------------------------------------------------------------
    # rendering items
    # ------------------------------------------------------------------
    def _items(self, folder: str, multiplier: float) -> List[dict]:
        sc = self.sample_config
        items = []
        for s in self.slots:
            items.append({
                "prompt": self.prompt_fn(s["prompt"]),
                "negative_prompt": s["neg"],
                "width": s["width"],
                "height": s["height"],
                "seed": s["seed"],
                "guidance_scale": s["guidance"],
                "guidance_rescale": sc.guidance_rescale,
                "num_inference_steps": s["steps"],
                "network_multiplier": multiplier,
                "output_path": os.path.join(folder, s["modality"], f"{s['id']}.png"),
                "output_ext": "png",
                "refiner_start_at": sc.refiner_start_at,
                "extra_values": sc.extra_values,
                "do_cfg_norm": sc.do_cfg_norm,
                "_renderer": s["renderer"],
                "_cache_key": f"identity:{s['id']}",
            })
        return items

    def step_dir(self, step: int) -> str:
        return os.path.join(self.root, f"step_{int(step):09d}")

    def is_eval_step(self, step: Optional[int], final: bool = False) -> bool:
        if not self.active or step is None or step <= 0:
            return False
        if final:
            return True
        every = max(1, int(self.sample_config.sample_every))
        # the pre-training sample (step 0/1) is covered by the baseline
        return step >= every and (step // every) % self.cfg.eval_every == 0

    def round_items(self, step: int) -> List[dict]:
        return self._items(self.step_dir(step), multiplier=float(self.sample_config.network_multiplier))

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _session(self, gpu_session):
        if gpu_session is not None and self.uses_gpu:
            with gpu_session():
                ids = {r: self.workers[r].send("to_gpu") for r in ROLES}
                for r in ROLES:
                    self.workers[r].wait(ids[r])
                try:
                    yield
                finally:
                    ids = {r: self.workers[r].send("to_cpu") for r in ROLES}
                    for r in ROLES:
                        self.workers[r].wait(ids[r])
        else:
            yield

    def _image_list(self, folder: str) -> Dict[str, List[dict]]:
        out = {r: [] for r in ROLES}
        for s in self.scored_slots:
            path = os.path.join(folder, s["modality"], f"{s['id']}.png")
            if os.path.exists(path):
                out[s["modality"]].append({"slot": s["id"], "path": path})
        return out

    def _score_images(self, images: Dict[str, List[dict]]) -> Dict[str, dict]:
        # workers on different cards score in parallel; on one card in turn
        same_card = self.gpus["face"] == self.gpus["body"] and self.uses_gpu
        results: Dict[str, dict] = {}
        if same_card:
            for r in ROLES:
                results.update(self._collect(self.workers[r].request("score", images=images[r])))
        else:
            ids = {r: self.workers[r].send("score", images=images[r]) for r in ROLES}
            for r in ROLES:
                results.update(self._collect(self.workers[r].wait(ids[r])))
        return results

    @staticmethod
    def _collect(reply: dict) -> Dict[str, dict]:
        return {res["slot"]: res for res in reply["results"]}

    def score_round(self, step: int, gpu_session):
        """Score a rendered round. Gpu workers score now while the trainer is
        unloaded; cpu workers score beside training and are picked up by poll()."""
        folder = self.step_dir(step)
        with open(os.path.join(folder, "DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        images = self._image_list(folder)
        if self.uses_gpu:
            print_acc(f"Identity eval: scoring step {step}")
            t0 = time.time()
            try:
                with self._session(gpu_session):
                    raw = self._score_images(images)
            except ChildError as e:
                print_acc(f"Identity eval: scoring failed for step {step}: {e}")
                return
            self._ready_metrics.append(self._finish_round(step, raw, time.time() - t0))
        else:
            ids = {r: self.workers[r].send("score", images=images[r]) for r in ROLES}
            self._pending.append({"step": step, "ids": ids, "t0": time.time(), "results": {}})

    def poll(self) -> List[dict]:
        """Finished rounds as [{'step', 'metrics'}], oldest first."""
        still = []
        for p in self._pending:
            try:
                for r, req in list(p["ids"].items()):
                    reply = self.workers[r].poll(req)
                    if reply is not None:
                        p["results"].update(self._collect(reply))
                        del p["ids"][r]
            except ChildError as e:
                print_acc(f"Identity eval: scoring failed for step {p['step']}: {e}")
                continue
            if p["ids"]:
                still.append(p)
            else:
                self._ready_metrics.append(self._finish_round(p["step"], p["results"], time.time() - p["t0"]))
        self._pending = still
        out, self._ready_metrics = self._ready_metrics, []
        return [m for m in out if m is not None]

    def finish(self, timeout_s: float) -> List[dict]:
        deadline = time.time() + timeout_s
        done = self.poll()
        while self._pending and time.time() < deadline:
            time.sleep(2.0)
            done.extend(self.poll())
        if self._pending:
            print_acc("Identity eval did not finish scoring step(s) "
                      + ", ".join(str(p["step"]) for p in self._pending))
        return done

    def _finish_round(self, step: int, raw: Dict[str, dict], seconds: float) -> Optional[dict]:
        a = self.anchors
        per_image, by_modality, over_target = {}, {r: [] for r in ROLES}, 0
        for s in self.scored_slots:
            role = s["modality"]
            weights = {f: 1.0 for f in self.features[role]}
            r = raw.get(s["id"], {"status": S.FAILED, "reason": "image missing", "raws": {}})
            entry = {"modality": role, "bin": s["bin"], "status": r["status"], "reason": r.get("reason"),
                     "score": None}
            if r["status"] == S.VALID:
                base = a["baselines"].get(s["id"])
                if base is None:
                    entry["status"] = S.UNCALIBRATED
                    entry["reason"] = "baseline image was not scorable"
                else:
                    res = S.score_slot(r["raws"], base, a["targets"], weights)
                    entry.update(res)
                    if res["max_ratio"] is not None and res["max_ratio"] > 1.0:
                        over_target += 1
            per_image[s["id"]] = entry
            by_modality[role].append({"bin": s["bin"], "score": entry["score"]})

        agg = {r: S.aggregate(by_modality[r], len(by_modality[r])) for r in ROLES}
        identity = S.fuse(agg["face"]["score"], agg["body"]["score"], self.cfg.face_weight, self.cfg.body_weight)
        warnings = []
        if over_target:
            warnings.append(f"{over_target} slot(s) past the reference target: possible memorization")
        if not self.cfg.holdout_references:
            warnings.append("no holdout references")
        summary = {
            "step": step,
            "calibration_id": self.calibration_id,
            "metric_versions": self.metric_versions,
            "face": agg["face"],
            "body": agg["body"],
            "identity": {"score": identity, "valid": identity is not None, "formula": "weighted_geomean_v1"},
            "over_target_slots": over_target,
            "warnings": warnings,
            "seconds": round(seconds, 1),
            "per_image_path": os.path.join(os.path.basename(self.step_dir(step)), "per_image.json"),
        }
        folder = self.step_dir(step)
        M.write_json(os.path.join(folder, "per_image.json"), per_image)
        M.write_json(os.path.join(folder, "checkpoint_summary.json"), summary)
        with open(os.path.join(self.root, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        def fmt(v):
            return "N/A" if v is None else f"{v:.1f}"
        print_acc(f"Identity @ step {step}: face {fmt(agg['face']['score'])} "
                  f"({agg['face']['coverage']:.0%}) | body {fmt(agg['body']['score'])} "
                  f"({agg['body']['coverage']:.0%}) | identity {fmt(identity)}  [{seconds:.0f}s]")
        for w in warnings[:1]:
            print_acc(f"Identity eval warning: {w}")
        metrics = {}
        if agg["face"]["score"] is not None:
            metrics["identity/face"] = agg["face"]["score"]
        if agg["body"]["score"] is not None:
            metrics["identity/body"] = agg["body"]["score"]
        if identity is not None:
            metrics["identity/overall"] = identity
        return {"step": step, "metrics": metrics}

    def close(self):
        for w in self.workers.values():
            w.close()
        self.workers = {}
        self.active = False
