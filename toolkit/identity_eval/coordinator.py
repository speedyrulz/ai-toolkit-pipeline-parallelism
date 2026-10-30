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
        self.flex_slots = [s for s in self.slots if s["modality"] == "flex"]
        self.model_fingerprint = model_fingerprint

        face_gpu = cfg.face_gpu if cfg.face_gpu is not None else trainer_gpu
        body_gpu = cfg.body_gpu if cfg.body_gpu is not None else (
            second_gpu if second_gpu is not None else trainer_gpu)
        self.gpus = {"face": face_gpu, "body": body_gpu}
        self.trainer_gpu = trainer_gpu
        self.second_gpu = second_gpu
        self.references = {
            "face": cfg.face_references or default_references,
            "body": cfg.body_references or default_references,
        }
        self.workers: Dict[str, JsonProcess] = {}
        self.features: Dict[str, List[str]] = {}
        self.weights: Dict[str, Dict[str, float]] = {}
        self.model_ids: Dict[str, dict] = {}
        self.metric_versions: Dict[str, str] = {}
        self.anchors: Optional[dict] = None
        self.calibration_id: Optional[str] = None
        self.active = False
        self._pending: List[dict] = []   # rounds scoring beside training
        self._ready_metrics: List[dict] = []
        self._helper_offloaded = None    # the sample helper, while background scoring holds its card
        self._training_wait_s = 0.0      # time sample rounds waited on background scoring

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    @property
    def uses_gpu(self) -> bool:
        return any(g is not None and g >= 0 for g in self.gpus.values())

    @property
    def background(self) -> bool:
        """Rounds score beside training when no scoring worker shares the
        trainer's card (cpu workers, or gpu workers on the second card)."""
        return all(g is None or g < 0 or g != self.trainer_gpu for g in self.gpus.values())

    def devices(self) -> Dict[str, str]:
        return {r: "cpu" if g is not None and g < 0 else f"gpu {g}" for r, g in self.gpus.items()}

    def _write_status(self, state: str, step: Optional[int] = None):
        """status.json for the UI: how rounds are scored (beside training or
        with training paused) and what scoring is doing right now."""
        M.write_json(os.path.join(self.root, "status.json"), {
            "mode": "background" if self.background else "paused",
            "devices": self.devices(),
            "trainer_gpu": self.trainer_gpu,
            "state": state,  # idle | scoring | waiting | stopped
            "step": step,
            "pending_steps": [p["step"] for p in self._pending],
            "training_wait_s": round(self._training_wait_s, 1),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def _worker_models(self, role: str) -> dict:
        models = dict(self.cfg.models)
        if role == "body" and self.cfg.smplx_model_dir:
            models.setdefault("smplx_model_dir", self.cfg.smplx_model_dir)
        return models

    def _start_workers(self):
        python = self.cfg.evaluator_python or sys.executable
        for role in ROLES:
            gpu = self.gpus[role]
            device = "cuda" if gpu is not None and gpu >= 0 else "cpu"
            where = f"gpu {gpu}" if device == "cuda" else "cpu"
            print_acc(f"Identity eval: starting {role} worker on {where}")
            w = JsonProcess(
                f"identity {role} worker",
                [python, "-u", WORKER_MAIN, "--role", role, "--device", device,
                 "--backend", self.cfg.backend, "--models", json.dumps(self._worker_models(role))],
                os.path.join(self.root, f"worker_{role}.log"),
                env=gpu_env(gpu),
            )
            # workers load their models before they answer (the first run downloads them)
            w.start()
            info = w.request("info", timeout=60)
            self.workers[role] = w
            self.features[role] = info["features"]
            self.metric_versions[role] = info["metric_version"]
            self.weights[role] = info.get("weights") or {f: 1.0 for f in info["features"]}
            self.model_ids[role] = info.get("models") or {}

    def _fingerprint(self) -> dict:
        return {
            "model": self.model_fingerprint,
            "slots": slots_signature(self.slots),
            "sampler": {
                "sampler": self.sample_config.sampler,
                "guidance_rescale": self.sample_config.guidance_rescale,
                "network_multiplier": self.sample_config.network_multiplier,
            },
            # extra sample LoRAs are part of every eval image, baseline included
            "sample_loras": [
                {"file": M.fingerprint_model(l["path"]), "strength": float(l.get("strength", 1.0))}
                for l in (getattr(self.sample_config, "loras", None) or [])
            ],
            "references": {r: M.hash_images(M.list_images([self.references[r]])) for r in ROLES},
            "metric_versions": self.metric_versions,
            "models": self.model_ids,
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
        self._write_status("idle")
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
            "reference_views": {r: prepared[r].get("views") for r in ROLES},
            "excluded_references": {r: prepared[r]["excluded"] for r in ROLES},
        }
        self.calibration_id = cal_id
        M.write_json(os.path.join(self.root, "anchors.json"), self.anchors)
        M.write_json(os.path.join(self.root, "manifest.json"), {
            "calibration_id": cal_id,
            "metric_versions": self.metric_versions,
            "features": self.features,
            "weights": self.weights,
            "models": self.model_ids,
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

    @staticmethod
    def flex_text(prompt: str) -> str:
        """Prompt-adherence text: the suite prompt without the identity token."""
        return " ".join(prompt.replace("[trigger]", "a person").split())

    def _image_list(self, folder: str) -> Dict[str, List[dict]]:
        out = {r: [] for r in ROLES}
        for s in self.scored_slots + self.flex_slots:
            path = os.path.join(folder, s["modality"], f"{s['id']}.png")
            if not os.path.exists(path):
                continue
            item = {"slot": s["id"], "path": path, "modality": s["modality"]}
            if s["modality"] == "flex":
                item["prompt"] = self.flex_text(s["prompt"])
            # the face worker owns the flex suite
            out["face" if s["modality"] == "flex" else s["modality"]].append(item)
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

    def score_round(self, step: int, gpu_session, helper=None):
        """Score a rendered round. When a worker shares the trainer's card it
        scores now while the trainer is unloaded; otherwise the round scores
        beside training (on cpu or the second card, with the sample helper
        moved off that card until the next sample round) and poll() picks it up."""
        folder = self.step_dir(step)
        with open(os.path.join(folder, "DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        images = self._image_list(folder)
        if self.background:
            self._score_background(step, images, helper)
        elif self.uses_gpu:
            print_acc(f"Identity eval: scoring step {step}")
            self._write_status("scoring", step)
            t0 = time.time()
            try:
                with self._session(gpu_session):
                    raw = self._score_images(images)
            except ChildError as e:
                print_acc(f"Identity eval: scoring failed for step {step}: {e}")
                self._write_status("idle")
                return
            self._ready_metrics.append(self._finish_round(step, raw, time.time() - t0))
            self._write_status("idle")

    def _score_background(self, step: int, images: Dict[str, List[dict]], helper):
        gpu_roles = [r for r in ROLES if self.gpus[r] is not None and self.gpus[r] >= 0]
        if (gpu_roles and helper is not None and self.second_gpu is not None
                and any(self.gpus[r] == self.second_gpu for r in gpu_roles)):
            helper.offload()
            self._helper_offloaded = helper
        try:
            ids, tail = {}, {}
            for r in ROLES:
                # each worker answers in order: load, score, unload
                if r in gpu_roles:
                    self.workers[r].send("to_gpu")
                ids[r] = self.workers[r].send("score", images=images[r])
                if r in gpu_roles:
                    tail[r] = self.workers[r].send("to_cpu")
        except ChildError as e:
            print_acc(f"Identity eval: scoring failed for step {step}: {e}")
            return
        where = ", ".join(sorted(set(self.devices().values())))
        print_acc(f"Identity eval: scoring step {step} on {where} beside training")
        self._pending.append({"step": step, "ids": ids, "tail": tail, "t0": time.time(), "results": {}})
        self._write_status("scoring", step)

    def _advance(self):
        still = []
        for p in self._pending:
            try:
                for r, req in list(p["ids"].items()):
                    reply = self.workers[r].poll(req)
                    if reply is not None:
                        p["results"].update(self._collect(reply))
                        del p["ids"][r]
                for r, req in list(p.get("tail", {}).items()):
                    if self.workers[r].poll(req) is not None:
                        del p["tail"][r]
            except ChildError as e:
                print_acc(f"Identity eval: scoring failed for step {p['step']}: {e}")
                continue
            if p["ids"] or p.get("tail"):
                still.append(p)
            else:
                self._ready_metrics.append(self._finish_round(p["step"], p["results"], time.time() - p["t0"]))
        finished = len(still) < len(self._pending)
        self._pending = still
        if finished:
            if still:
                self._write_status("scoring", still[0]["step"])
            else:
                self._write_status("idle")

    def poll(self) -> List[dict]:
        """Finished rounds as [{'step', 'metrics'}], oldest first."""
        self._advance()
        out, self._ready_metrics = self._ready_metrics, []
        return [m for m in out if m is not None]

    def drain(self, timeout_s: float = 1800.0):
        """Before a sample round: wait for background scoring to release the
        second card, then give it back to the sample helper. Finished rounds
        stay queued for the next poll()."""
        t0 = time.time()
        deadline = t0 + timeout_s
        if self._pending:
            print_acc("Identity eval: waiting for background scoring to finish")
            self._write_status("waiting", self._pending[0]["step"])
        waited = bool(self._pending)
        while self._pending and time.time() < deadline:
            self._advance()
            if self._pending:
                time.sleep(0.5)
        if waited:
            self._training_wait_s += time.time() - t0
            self._write_status("scoring" if self._pending else "idle",
                               self._pending[0]["step"] if self._pending else None)
        if self._helper_offloaded is not None:
            self._helper_offloaded.onload()
            self._helper_offloaded = None

    def finish(self, timeout_s: float) -> List[dict]:
        deadline = time.time() + timeout_s
        done = self.poll()
        while self._pending and time.time() < deadline:
            time.sleep(1.0)
            done.extend(self.poll())
        if self._pending:
            print_acc("Identity eval did not finish scoring step(s) "
                      + ", ".join(str(p["step"]) for p in self._pending))
        rows = self._write_ranking()
        if rows and rows[0]["eligible"]:
            print_acc(f"Identity eval: best checkpoint step {rows[0]['step']} "
                      f"(identity {rows[0]['identity']:.1f})")
        elif rows:
            print_acc("Identity eval: no round met the checkpoint ranking gates "
                      f"(face >= {self.cfg.rank_min_face:g}, body >= {self.cfg.rank_min_body:g}, "
                      f"flex >= -{self.cfg.flex_max_drop:g})")
        return done

    def _finish_round(self, step: int, raw: Dict[str, dict], seconds: float) -> Optional[dict]:
        a = self.anchors
        per_image, by_modality, over_target = {}, {r: [] for r in ROLES}, 0
        pooled = {r: self._pooled_baseline(r) for r in ROLES}
        for s in self.scored_slots:
            role = s["modality"]
            weights = self.weights[role]
            r = raw.get(s["id"], {"status": S.FAILED, "reason": "image missing", "raws": {}})
            # slots without a declared view are binned by the view the worker saw
            seen = (r.get("extra") or {}).get("view")
            entry = {"modality": role, "bin": seen if s["bin"] == "any" and seen else s["bin"],
                     "status": r["status"], "reason": r.get("reason"), "score": None, "extra": r.get("extra")}
            if r["status"] == S.VALID:
                base = a["baselines"].get(s["id"])
                if base is None:
                    # the base model missed this slot at step 0 (e.g. no face):
                    # fall back to the suite's median baseline, flagged
                    base = pooled[role]
                    entry["baseline"] = "pooled" if base else None
                else:
                    entry["baseline"] = "paired"
                if not base:
                    entry["status"] = S.UNCALIBRATED
                    entry["reason"] = "no scorable baseline image in this suite"
                else:
                    res = S.score_slot(r["raws"], base, a["targets"], weights)
                    entry.update(res)
                    if role == "body":
                        entry["confidence"] = self._agreement(res["features"])
                    if res["max_ratio"] is not None and res["max_ratio"] > 1.0:
                        over_target += 1
            per_image[s["id"]] = entry
            by_modality[role].append({"bin": entry["bin"], "score": entry["score"]})

        flex = self._flex(raw, per_image)
        agg = {r: S.aggregate(by_modality[r], len(by_modality[r])) for r in ROLES}
        identity = S.fuse(agg["face"]["score"], agg["body"]["score"], self.cfg.face_weight, self.cfg.body_weight)
        warnings = []
        if over_target:
            warnings.append(f"{over_target} slot(s) past the reference target: possible memorization")
        low = [k for k, v in per_image.items() if v.get("confidence") == "LOW"]
        if low:
            warnings.append(f"body estimators disagree on {len(low)} slot(s) (low confidence): {', '.join(low)}")
        pooled_slots = [k for k, v in per_image.items() if v.get("baseline") == "pooled"]
        if pooled_slots:
            warnings.append(f"{len(pooled_slots)} slot(s) use the suite's median step-0 baseline "
                            f"(their own step-0 image was not scorable): {', '.join(pooled_slots)}")
        if flex["delta"] is not None and flex["delta"] < -self.cfg.flex_max_drop:
            warnings.append(f"prompt adherence {flex['delta']:+.1f} vs step 0 (flex suite): "
                            "the LoRA may be overriding prompts")
        if not self.cfg.holdout_references:
            warnings.append("no holdout references")
        summary = {
            "step": step,
            "calibration_id": self.calibration_id,
            "metric_versions": self.metric_versions,
            "face": agg["face"],
            "body": agg["body"],
            "identity": {"score": identity, "valid": identity is not None, "formula": "weighted_geomean_v1"},
            "flex": flex,
            "over_target_slots": over_target,
            "warnings": warnings,
            "seconds": round(seconds, 1),
            "scored": "background" if self.background else "paused",
            "devices": self.devices(),
            "per_image_path": os.path.join(os.path.basename(self.step_dir(step)), "per_image.json"),
        }
        folder = self.step_dir(step)
        M.write_json(os.path.join(folder, "per_image.json"), per_image)
        M.write_json(os.path.join(folder, "checkpoint_summary.json"), summary)
        with open(os.path.join(self.root, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        self._write_ranking()

        def fmt(v):
            return "N/A" if v is None else f"{v:.1f}"
        print_acc(f"Identity @ step {step}: face {fmt(agg['face']['score'])} "
                  f"({agg['face']['coverage']:.0%}) | body {fmt(agg['body']['score'])} "
                  f"({agg['body']['coverage']:.0%}) | identity {fmt(identity)} | flex "
                  f"{'N/A' if flex['delta'] is None else format(flex['delta'], '+.1f')}  [{seconds:.0f}s]")
        for w in warnings[:1]:
            print_acc(f"Identity eval warning: {w}")
        metrics = {}
        if agg["face"]["score"] is not None:
            metrics["identity/face"] = agg["face"]["score"]
        if agg["body"]["score"] is not None:
            metrics["identity/body"] = agg["body"]["score"]
        if identity is not None:
            metrics["identity/overall"] = identity
        if flex["delta"] is not None:
            metrics["identity/flex"] = flex["delta"]
        return {"step": step, "metrics": metrics}

    def _write_ranking(self) -> List[dict]:
        """ranking.json: every scored round ranked for checkpoint selection
        (the UI joins it with the checkpoint files on disk)."""
        by_step = {}
        try:
            with open(os.path.join(self.root, "results.jsonl"), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        by_step[r["step"]] = r  # a resumed run may rescore a step
        except (OSError, ValueError):
            pass
        c = self.cfg
        rows = S.rank_rounds(list(by_step.values()), c.rank_min_face, c.rank_min_body, c.flex_max_drop)
        M.write_json(os.path.join(self.root, "ranking.json"), {
            "criteria": {"min_face": c.rank_min_face, "min_body": c.rank_min_body,
                         "flex_max_drop": c.flex_max_drop},
            "rows": rows,
        })
        return rows

    @staticmethod
    def _agreement(features: dict) -> Optional[str]:
        """Confidence tag from how closely the two body estimators' calibrated
        shape scores agree (MHR via SAM 3D Body, SMPL-X via SMPLest-X)."""
        a = (features.get("body_shape_mhr") or {}).get("score")
        b = (features.get("body_shape_smplx") or {}).get("score")
        if a is None or b is None:
            return None
        d = abs(a - b)
        return "HIGH" if d < 15 else "MEDIUM" if d < 30 else "LOW"

    def _pooled_baseline(self, role: str) -> Dict[str, float]:
        """Per-feature median of the paired baselines of one suite."""
        rows = [self.anchors["baselines"][s["id"]] for s in self.scored_slots
                if s["modality"] == role and s["id"] in self.anchors["baselines"]]
        return {f: S.median(r.get(f) for r in rows) for f in self.features[role]
                if any(r.get(f) is not None for r in rows)}

    def _flex(self, raw: Dict[str, dict], per_image: dict) -> dict:
        """Prompt adherence vs step 0: mean SigLIP cosine change over flex
        slots, x100 (points; negative = the prompts are followed less)."""
        base = self.anchors["baselines"]
        deltas = []
        for s in self.flex_slots:
            r = raw.get(s["id"], {"status": S.FAILED, "reason": "image missing", "raws": {}})
            now = r.get("raws", {}).get("flex_siglip")
            then = (base.get(s["id"]) or {}).get("flex_siglip")
            d = None if now is None or then is None else 100.0 * (now - then)
            per_image[s["id"]] = {"modality": "flex", "bin": s["bin"], "status": r["status"],
                                  "reason": r.get("reason"), "raw": now, "baseline": then, "delta": d}
            if d is not None:
                deltas.append(d)
        return {
            "delta": sum(deltas) / len(deltas) if deltas else None,
            "valid_slots": len(deltas),
            "slots": len(self.flex_slots),
        }

    def close(self):
        if self.active:
            self._write_status("stopped")
        for w in self.workers.values():
            w.close()
        self.workers = {}
        self.active = False
