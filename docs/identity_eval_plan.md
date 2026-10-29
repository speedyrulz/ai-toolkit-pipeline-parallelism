# Identity Eval: implementation plan

Revision of *LoRA Character Identity Consistency System v1.0* (29 Oct 2026) for
this fork. Everything in the source spec stands unless changed here.

## Decisions (agreed)

| # | Decision | Change from spec |
|---|----------|------------------|
| D1 | **Per-slot paired baseline.** `B_j,s` = raw value of slot `s` at step 0. Global-max baseline kept as a secondary "strict" readout. | Replaces `B_j = max(step0)` as the official anchor |
| D2 | **Raw-over-target is logged** (e.g. face at 140% of target) and flagged as a memorization warning. Headline scores stay clipped to [0,100]. | Addition |
| D3 | **References default to the training images**; a holdout bank is optional and its absence shows a warning. | Spec assumed a holdout |
| D4 | **Prompt-flexibility guard**: a FLEX_EVAL suite (varied outfit/setting/pose) scored for prompt adherence (SigLIP text-image) relative to step 0. Not folded into Identity; used in checkpoint selection guardrails. | Addition |
| D5 | **Standalone evaluator venv** managed by the toolkit (Windows or WSL), not ComfyUI. | Spec left env open |
| D6 | **Direct in-process hook** in `BaseSDTrainProcess.sample()` plus a standalone CLI over the same worker. Atomic `DONE` marker per round kept. | Replaces filesystem watcher as the primary path |
| D7 | **SMPL-X kept in v1** alongside MHR (see Body below). | Was proposed for removal; kept |
| D8 | **Reduced v1 model set**; MICA, MediaPipe, SMIRK, Sapiens2 normals/pointmap come later. Each feature-set change is a new `metric_version`. | Phasing |
| D9 | **Evaluator venv is Windows native.** SMPL-X files live in `C:\SMPL-X\`. Use is noncommercial and non-pornographic, so InsightFace, SMPL-X and MICA licenses are acceptable. | Settled prerequisites |
| D10 | **Identity eval requires a non-sharded run** for now; config validation rejects `identity_eval.enabled` with `pipeline_sharding`, and the UI card is disabled. | Restriction |
| D11 | **Second GPU for sampling and scoring** in non-sharded runs; the training model is **unloaded from the GPUs while scoring runs** and reloaded after (see "Two-GPU execution"). | Addition |

## Architecture

```
trainer (BaseSDTrainProcess)
  sample() ─┬─ regular samples (unchanged)
            └─ IdentityEval.render_round(step)      # FACE/BODY/FLEX suites, fixed seeds
                 └─ writes identity_eval/step_XXXXXXXXX/{face,body,flex}/ + DONE
                 └─ submit(step) ──stdin JSON──▶ worker process (evaluator venv)
  train loop ── poll() ◀──stdout JSON── results ──▶ logger: identity/* metrics
  end ───────── finish(wait_at_end_minutes)

worker (toolkit/identity_eval/worker/, own venv; CPU or chosen GPU)
  startup: load/verify calibration (hash check) or build it
  per round: face pipeline, body pipeline, flex pipeline → calibrated scorer
  writes per_image.json, checkpoint_summary.json, overlays/
```

Reused from the removed likeness code: subprocess launcher with
JSON-over-stdio protocol, WSL path translation, non-blocking poll, end-of-run
wait, prompt pre-encoding for when text encoders are unloaded (`SDTrainer`
`sample_prompts_cache` pattern, swapped in per suite).

### Step 0 baseline

- Rendered with `network_multiplier=0`, independent of `skip_first_sample`.
- Stored in `identity_eval/baseline/` with a manifest of hashes (base model,
  prompts, seeds, sampler settings, evaluator version).
- On resume: anchors are reloaded, never recomputed. A hash mismatch blocks
  scoring with a visible `CALIBRATION_INVALID` warning.

### Output layout (`<save_root>/identity_eval/`)

```
manifest.json            run manifest (spec §2.3) + calibration_id
calibration/             reference templates, B/T anchors, LOO distributions
baseline/{face,body,flex}/
step_XXXXXXXXX/{face,body,flex}/ + DONE
step_XXXXXXXXX/per_image.json, checkpoint_summary.json, overlays/
results.jsonl            one summary per checkpoint (spec §8.1 schema)
worker.log
```

## Two-GPU execution (D11)

Applies to non-sharded runs: the trainer and its model live on `cuda:0`, and
`cuda:1` is otherwise idle. Pipelined sampling across shards is deferred.

### Per-round sequence

```
training pauses at the sample step
 1. render   cuda:0: trainer's own model (live LoRA)
             cuda:1: sample helper process (same base model, LoRA synced)
 2. unload   trainer model -> CPU RAM; helper model -> CPU RAM; empty_cache
 3. score    face+flex worker on cuda:0, body worker on cuda:1
             (worker models move CPU -> GPU, score, move back)
 4. reload   trainer model -> cuda:0; helper model -> cuda:1
training resumes; scores are logged against the round's step
```

Rounds without identity eval (`eval_every` > 1) do steps 1 and 4 only, with
no unload.

### Sample helper (cuda:1)

A separate process, started once per run, so both GPUs render truly in
parallel without sharing the trainer's global state (LoRA multiplier, hooks,
adapters).

- Loads the base model with the job's own `ModelConfig` through the toolkit's
  regular model loader, plus its own text encoders (so it does not depend on
  the trainer's prompt-embedding cache).
- Each round the trainer writes a LoRA snapshot (`network.save_weights` to
  `identity_eval/tmp/`, a few MB to a few hundred MB) and sends the helper its
  share of `GenerateImageConfig`s as JSON. The helper loads the snapshot into
  its own network and renders.
- **Fixed slot-to-device assignment for the whole run, step 0 included.**
  The two cards (4060 Ti / 5060 Ti) are different architectures and can give
  slightly different images for the same seed; a slot that always renders on
  the same card keeps the paired baseline (D1) valid.
- Splits every sample round (regular samples and eval suites), configured by
  `sample.helper_gpu: <physical index>`; works without identity eval too.
- Falls back to rendering everything on `cuda:0`, with a log line, when the
  job uses something the helper does not reproduce (custom adapters, refiner,
  sample LoRAs, sharding) or when the helper fails to start.

### Unload / reload of the training model

- Moves the frozen base model (UNet/transformer, text encoders, VAE) to CPU
  RAM and calls `torch.cuda.empty_cache()`, so each card is close to empty
  for scoring.
- LoRA parameters and optimizer state stay on the GPU (small, and moving
  8-bit optimizer state is fragile). Measured in the P0 spike; if they block
  scoring they move too.
- Reload restores every module to its original device and dtype and asserts
  it; a failed reload stops the run with a clear error instead of training on
  a half-loaded model.
- Quantized base models (qfloat8 etc.) are a known risk for CPU round trips;
  P0 verifies them or marks them unsupported for identity eval.

### Scoring workers

Two worker processes split **by modality**, each owning one card while the
trainer is unloaded:

| Worker | Models | Device |
|--------|--------|--------|
| face + flex | SCRFD, AntelopeV2, CVLFace, SigLIP | `cuda:0` |
| body | SAM 3D Body, SMPLest-X, SMPL-X fit | `cuda:1` |

Devices are configurable (`face_gpu`, `body_gpu`: physical GPU index, `-1` = CPU,
unset = trainer GPU for face, helper GPU for body).
With one GPU both workers use `cuda:0` in turn. Between rounds worker models
sit in pinned CPU RAM.

CPU scoring (`face_gpu`/`body_gpu: -1`) skips the unload and runs
concurrently with training instead, for runs where pausing is unwanted.

Calibration (reference templates and the step-0 baseline) runs at train start,
after the model loads and before the first sample, using the same unload sequence.

## Face (v1)

SCRFD detect → main-subject selection → align →
**AntelopeV2** (ArcFace R100) + **CVLFace KP-RPE** embeddings →
view-binned prototypes (front / ¾ / profile) → LOO target per feature.

v1 face score: `F = 0.6*F_CVL + 0.4*F_Arc` (`metric_version 1.0.0-face-v1`).
Spec weights (adds MICA 25%, landmarks 10%) arrive in v2 as `2.0.0`.

## Body (v1, with SMPL-X)

1. Person detect + mask; reject truncation / multi-person.
2. **SAM 3D Body** → MHR mesh, pose, camera per image.
3. **SMPLest-X** → SMPL-X params per image (independent second estimate).
4. **Canonical reference fit in SMPL-X**: one shared `betas` across all good
   reference views, per-view pose/camera, initialized from SMPLest-X and from
   MHR→SMPL-X conversion (official tool); SMPLify-X-style optimization against
   2D keypoints + silhouettes.
5. Head excluded from all body measurements.

Body features, each calibrated independently:

| Feature | How | Weight |
|---------|-----|--------|
| Canonical mesh shape | pose-neutralized vertex distance, computed **separately** in MHR and SMPL-X space and calibrated separately; the two calibrated scores averaged. Never compare MHR params with SMPL-X betas. | 40% |
| Reprojected silhouette | reference canonical shape posed with the *generated* image's estimated pose/camera, rendered, IoU + contour chamfer vs the generated mask | 30% |
| Proportions | anthropometrics from the SMPL-X mesh (shoulder/hip width, waist/hip, torso/limb lengths, circumferences) | 30% |
| Surface detail | deferred to v3 (Sapiens2 normals) | — |

v1 body score is `metric_version 1.0.0-body-v1` with the 40/30/30 weights.

**Estimator agreement** (MHR vs SMPL-X shape, after conversion) feeds the
LOW/MEDIUM/HIGH confidence tag, not the score.

Rendering silhouettes uses a CPU triangle rasterizer (projected faces via
OpenCV `fillPoly`) to avoid a PyTorch3D build on the RTX 5060 Ti (sm_120).

## Calibration and fusion

Per feature `j`, per slot `s`:

```
B_js = raw_j(step0 image of slot s)
T_j  = q75( raw_j(r_i, template_without_r_i) for r_i in references )
if T_j <= B_js + eps: slot s, feature j = NOT_CALIBRATABLE
score_js = 100 * clip((raw_j - B_js) / (T_j - B_js), 0, 1)
over_target_js = (raw_j - B_js) / (T_j - B_js)      # logged unclipped (D2)
```

Aggregation, modality fusion, statuses and coverage gates follow spec
§5.3–§6.3 unchanged: median per pose bin → macro-average; weighted geometric
mean `w_face=0.55, w_body=0.45`; missing modality → Identity N/A.

## Flexibility guard (D4)

FLEX_EVAL prompts vary outfit, setting and pose. Score = SigLIP text-image
similarity vs the same slot at step 0, reported as `identity/flex` (delta,
not 0–100). Checkpoint ranking requires F ≥ 70, B ≥ 70 and flex not below
a configurable drop from baseline.

## Config (YAML, `process[0].identity_eval`)

```yaml
identity_eval:
  enabled: true
  eval_every: 1                  # in sample rounds
  evaluator_python: ""           # empty = toolkit-managed venv (Windows native)
  face_gpu: null                 # physical index; -1 = cpu; null = trainer gpu
  body_gpu: null                 # physical index; -1 = cpu; null = helper gpu
# sample.helper_gpu: 1           # second GPU renders through the sample helper
  face_references: path/to/face_refs      # default: dataset folder
  body_references: path/to/body_refs      # default: dataset folder
  holdout_references: ""
  smplx_model_dir: C:\SMPL-X              # user-downloaded SMPL-X model files
  face_suite: [ {prompt, seed?, width?, height?} ... ]
  body_suite: [ ... ]
  flex_suite: [ ... ]
  wait_at_end_minutes: 30
```

## UI

- **Job form**: "Identity Eval" card (paths, device, suites with sensible
  default prompts, `eval_every`).
- **Loss graph**: `identity/face`, `identity/body`, `identity/overall` on a
  fixed 0–100 axis; `identity/flex` on its own axis.
- **Job page "Identity" tab**: spec §8.2 wireframe — three headline numbers
  with coverage/status, per-angle breakdown, thumbnails per suite per step,
  raw values + anchors + QC reasons, memorization and flex warnings.

## Status

P0 is implemented with a placeholder scorer (`metric_version 0.0.0-stub`,
grayscale-thumbnail similarity) so the pipeline runs end to end. Still open
from P0: the model-install spike (SAM 3D Body, SMPLest-X, CVLFace on Windows,
CUDA 12.8+ for sm_120) and timing the unload/reload on a real run.

## Phases

| Phase | Ships | Exit criterion |
|-------|-------|----------------|
| P0 | Config, suites, step-0 baseline, manifest/hashes, worker protocol with a stub scorer (two workers, device routing), sample helper on the second GPU, trainer unload/reload, sharding validation, results store, UI card + empty Identity tab, scoring-math unit tests (spec §9.1). Spikes: model installs, unload/reload timing | Baseline locks; resume reloads anchors; deterministic replay; unload/reload leaves training numerically unchanged |
| P1 | Face v1 + flex guard | Face 0–100 bounded; cross-pose cases reviewed |
| P2 | Body v1 (MHR + SMPL-X canonical fit, silhouette, proportions) | Front/side/back comparisons credible; estimator agreement reported |
| P3 | Identity fusion, graph series, Identity tab complete, checkpoint ranking | Full run without blocking the trainer |
| v2+ | MICA, MediaPipe landmarks, SMIRK, Sapiens2 masks/normals | Each gated on holdout improvement; new metric_version |

## Prerequisites from the user

- Download from smpl-x.is.tue.mpg.de (license acceptance required) into
  `C:\SMPL-X\`:
  - **SMPL-X v1.1 (NPZ+PKL)**: the body model. Only the neutral model is used.
    Not the "removed head bun" versions: their shape space differs from v1.1,
    which SMPLest-X is trained on, and all SMPL-X shapes must share one space.
  - **VPoser v2.0**: pose prior for the multi-view canonical fit.
  - **smplx_parts_segm.pkl**: per-triangle body-part labels (head exclusion,
    regional measurements).
  - **SMPL-X MANO FLAME Correspondences**: exact head/face vertex indices.

  Expected layout (what the `smplx` loader looks for):
  ```
  <smplx_model_dir>/smplx/SMPLX_NEUTRAL.npz
  <smplx_model_dir>/vposer_v2_05/...
  <smplx_model_dir>/smplx_parts_segm.pkl
  <smplx_model_dir>/SMPL-X__FLAME_vertex_ids.npy, MANO_SMPLX_vertex_ids.pkl
  ```

## Risks to check early (P0 spike)

- SMPLest-X and SAM 3D Body install on Windows native with a CUDA 12.8+
  PyTorch for sm_120 (RTX 5060 Ti).
- Per-round pause cost: render + unload + score + reload.
- Unload/reload round trip with quantized base models.
- Same-seed output differences between the two cards (confirms fixed slot assignment is needed).
- Multi-view SMPL-X fit stability with 6–12 references.
