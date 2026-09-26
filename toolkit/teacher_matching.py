"""Teacher-matching training math (MiniMax-H3 and any joint flow model).

A frozen base ("teacher": the same transformer with the LoRA disabled) runs
once per step under privileged conditions the student never sees at
inference; its prediction replaces the flow target. These helpers shape that
loss and the timestep draw. They are plain tensor functions with no trainer
state so they can be unit tested on CPU.

The algorithm and the four mechanisms below (DC attenuation, the decomposed
magnitude/direction loss, the preservation-anchor density compensation and
the timestep focus mixture) follow kohya-ss/musubi-tuner's MiniMax-H3
teacher matching (``src/musubi_tuner/minimax_h3_train_network.py``,
Apache-2.0); the implementations here are per-sample rewrites for
ai-toolkit's ``(B, ...)`` loss pipeline.
"""

from typing import Dict, Optional, Tuple

import torch


TEACHER_CONDITIONS_REF = "ref"
TEACHER_CONDITIONS_SUBJECT_REF = "subject_ref"
TEACHER_CONDITIONS = (TEACHER_CONDITIONS_REF, TEACHER_CONDITIONS_SUBJECT_REF)

# the marker the trainer sets on the batch for the anchor pass (LoRA off, the
# student's own text and layout, no conditions)
TEACHER_PASS_ANCHOR = "anchor"

# validated upper gate for the complete-information (ref) teacher: above it the
# conditioned content is unpredictable from the text, so the band is better
# spent as a base-preservation anchor
SIGMA_MAX_RECOMMENDED_COMPLETE_INFORMATION = 0.75


def normalize_teacher_conditions(value) -> str:
    parts = [p.strip() for p in str(value).split(",")]
    if parts == [TEACHER_CONDITIONS_REF]:
        return TEACHER_CONDITIONS_REF
    if parts == [TEACHER_CONDITIONS_SUBJECT_REF]:
        return TEACHER_CONDITIONS_SUBJECT_REF
    raise ValueError(
        "teacher_conditions must be one of "
        f"{', '.join(repr(c) for c in TEACHER_CONDITIONS)}, got {value!r}"
    )


# ---------------------------------------------------------------------------
# Sigma / timestep helpers
# ---------------------------------------------------------------------------


def shift_sigma(sigma, shift: float):
    """Exponential timeshift: ``shift * s / (1 + (shift - 1) * s)``."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def unshift_sigma(sigma, shift: float):
    """Inverse of :func:`shift_sigma`: the pre-shift base draw behind a
    shifted sigma."""
    return sigma / (shift + sigma * (1.0 - shift))


def base_sigma_from_uniform(
    u: torch.Tensor,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    focus_min: float = 0.0,
    focus_max: float = 1.0,
    focus_prob: float = 0.0,
) -> torch.Tensor:
    """Deterministic map of a uniform [0, 1) draw onto the training base sigma.

    With probability ``focus_prob`` the sample lands uniformly in the focus
    band ``[focus_min, focus_max)``; otherwise it is uniform over the clipped
    range ``[lower, upper)``. The band density becomes
    ``prob + (1 - prob) * (band width / range width)``, so the rest of the
    range (including a preservation-anchor band) keeps nonzero coverage.
    """
    passthrough = lower + (upper - lower) * u
    if focus_prob <= 0.0:
        return passthrough
    focused = focus_min + (focus_max - focus_min) * (u / focus_prob)
    passthrough = lower + (upper - lower) * (
        (u - focus_prob) / max(1.0 - focus_prob, 1e-8)
    )
    return torch.where(u < focus_prob, focused, passthrough)


# ---------------------------------------------------------------------------
# Loss shaping
# ---------------------------------------------------------------------------


def _non_batch_channel_dims(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(range(2, x.ndim))


def dc_attenuated_prediction(
    pred: torch.Tensor, target: torch.Tensor, dc_weight: float
) -> torch.Tensor:
    """Scale the residual's per-channel DC so that
    ``mse(pred', target) = mse_ac + dc_weight * mse_dc``.

    The DC of the residual is a global colour/tone cast (the style axis);
    attenuating it in the loss stops coherent palette absorption without
    touching the spatially structured AC content. A linear map of the
    residual, so gradients stay exact.
    """
    residual = pred - target
    residual_dc = residual.mean(dim=_non_batch_channel_dims(residual), keepdim=True)
    return pred - (1.0 - dc_weight**0.5) * residual_dc


def decomposed_flow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mag_weight: float,
    dir_weight: float = 1.0,
) -> torch.Tensor:
    """Per-sample magnitude/direction split of the MSE with the norm-shrinkage
    coupling removed. Returns a ``(B,)`` tensor.

    Exact identity: ``||p - t||^2 = (||p|| - ||t||)^2 + 2 ||p|| ||t|| (1 - cos)``.
    In plain MSE the direction term's ``||p||`` factor couples the two
    components: hedging the direction pays off by shrinking the norm, which
    drives the prediction toward the conditional mean's reduced magnitude
    (the wash-out). Detaching ``||p||`` in the direction term makes its
    gradient purely rotational, so the magnitude optimum becomes
    ``E[||t||]`` (full per-sample commitment) instead of ``||E[t]||``. At
    unit weights the loss VALUE still equals the per-sample MSE exactly.
    """
    b = pred.shape[0]
    p = pred.reshape(b, -1)
    t = target.reshape(b, -1)
    p_norm = p.norm(dim=1)
    t_norm = t.norm(dim=1)
    eps = 1e-12
    cos = (p * t).sum(dim=1) / (p_norm * t_norm + eps)
    magnitude_term = (p_norm - t_norm).pow(2)
    direction_term = 2.0 * p_norm.detach() * t_norm * (1.0 - cos)
    return (mag_weight * magnitude_term + dir_weight * direction_term) / p.shape[1]


def preservation_density_compensation(
    sigma_max: float,
    focus_min: float,
    focus_max: float,
    focus_prob: float,
    sigma_min: float = 0.0,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    """Loss-weight correction that keeps the preservation anchor's expected
    gradient share invariant under timestep focus.

    Focus concentrates the base-sigma draw on the teaching band and thins the
    anchor bands (base sigma > ``sigma_max``, and < ``sigma_min`` when a lower
    gate is set) from their uniform share of the clipped range to
    ``(1 - p) * uniform + p * overlap / (max - min)``; multiplying each anchor
    step's loss by ``uniform / focused`` restores the anchor's per-unit-time
    pull.
    """

    def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    anchor_width = overlap(sigma_max, 1.0, lower, upper) + overlap(
        0.0, sigma_min, lower, upper
    )
    if anchor_width <= 0.0 or focus_prob <= 0.0:
        return 1.0
    uniform_share = anchor_width / (upper - lower)
    focus_overlap = overlap(sigma_max, 1.0, focus_min, focus_max) + overlap(
        0.0, sigma_min, focus_min, focus_max
    )
    focused_share = (1.0 - focus_prob) * uniform_share + focus_prob * focus_overlap / (
        focus_max - focus_min
    )
    if focused_share <= 0.0:
        # the anchor band is never sampled, so the multiplier is never applied
        return 1.0
    return uniform_share / focused_share


def prediction_geometry_log(
    label: str, prediction: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    """Cosine, norm ratio and the residual DC/AC split between prediction and
    target (batch-averaged scalars for the step logs).

    ``cos`` isolates the direction component of the residual; ``norm_ratio``
    (student / target, 1 = matched) the magnitude component, and drifting
    above 1 is an early warning for burn-style amplification. The residual is
    split into its per-channel mean (DC: a global tone cast, the style
    component) and the remainder (AC: structured content), so a shrinking gap
    reads as style or content being learned.
    """
    student = prediction.detach().float()
    reference = target.detach().float()
    b = student.shape[0]
    s = student.reshape(b, -1)
    r = reference.reshape(b, -1)
    eps = 1e-12
    s_norm = s.norm(dim=1)
    r_norm = r.norm(dim=1)
    cos = (s * r).sum(dim=1) / (s_norm * r_norm + eps)
    residual = student - reference
    residual_dc = residual.mean(dim=_non_batch_channel_dims(residual), keepdim=True)
    residual_ac = residual - residual_dc
    return {
        f"teacher/{label}_cos": cos.mean().item(),
        f"teacher/{label}_norm_ratio": (s_norm / (r_norm + eps)).mean().item(),
        f"teacher/{label}_residual_dc_rms": residual_dc.pow(2).mean().sqrt().item(),
        f"teacher/{label}_residual_ac_rms": residual_ac.pow(2).mean().sqrt().item(),
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TeacherMatchingConfig:
    """The teacher-matching knobs a model parses out of ``model_kwargs``.

    ``dopsd: true`` is accepted as the legacy spelling of
    ``teacher_matching: true`` + ``teacher_conditions: ref`` with the
    upstream token-style teacher caption and the bleed term on.
    """

    def __init__(self, kw: dict):
        legacy_dopsd = bool(kw.get("dopsd", False))
        self.enabled = bool(kw.get("teacher_matching", False)) or legacy_dopsd
        self.conditions: Optional[str] = None
        if not self.enabled:
            return
        self.conditions = normalize_teacher_conditions(
            kw.get("teacher_conditions", TEACHER_CONDITIONS_REF)
        )
        # how the teacher caption is built: "musubi" wraps the caption in the
        # official declaration blocks; "token" is upstream D-OPSD's trigger
        # word -> <Picture 1>/<Video 1> substitution
        self.caption_style = str(
            kw.get("teacher_caption_style", "token" if legacy_dopsd else "musubi")
        ).lower()
        if self.caption_style not in ("musubi", "token"):
            raise ValueError(
                f"teacher_caption_style must be 'musubi' or 'token', got {self.caption_style!r}"
            )
        if self.caption_style == "token" and self.conditions != TEACHER_CONDITIONS_REF:
            raise ValueError("teacher_caption_style 'token' is only defined for the 'ref' teacher")
        self.sigma_min = float(kw.get("teacher_condition_sigma_min", 0.0))
        self.sigma_max = float(kw.get("teacher_condition_sigma_max", 1.0))
        if not 0.0 <= self.sigma_max <= 1.0:
            raise ValueError(f"teacher_condition_sigma_max must be in [0, 1], got {self.sigma_max}")
        if not 0.0 <= self.sigma_min <= self.sigma_max:
            raise ValueError(
                f"teacher_condition_sigma_min must be in [0, teacher_condition_sigma_max], got {self.sigma_min}"
            )
        self.dc_weight = float(kw.get("teacher_loss_dc_weight", 1.0))
        self.mag_weight = float(kw.get("teacher_loss_mag_weight", 1.0))
        self.preservation_weight = float(kw.get("teacher_preservation_weight", 1.0))
        for name, value in (
            ("teacher_loss_dc_weight", self.dc_weight),
            ("teacher_loss_mag_weight", self.mag_weight),
            ("teacher_preservation_weight", self.preservation_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        # D-OPSD bleed: extra loss toward the plain flow target, scaled to the
        # teacher loss magnitude. Not part of musubi's recipe; on by default
        # only for the legacy dopsd spelling
        self.bleed_strength = float(
            kw.get("dopsd_bleed_strength", kw.get("teacher_bleed_strength", 1.0 if legacy_dopsd else 0.0))
        )
        self.focus_prob = float(kw.get("timestep_focus_prob", 0.0))
        self.focus_min = float(kw.get("timestep_focus_min", 0.4))
        self.focus_max = float(kw.get("timestep_focus_max", 0.8))
        if not 0.0 <= self.focus_prob <= 1.0:
            raise ValueError(f"timestep_focus_prob must be in [0, 1], got {self.focus_prob}")
        if self.focus_prob > 0.0 and not (0.0 <= self.focus_min < self.focus_max <= 1.0):
            raise ValueError(
                "timestep_focus_min/max must satisfy 0 <= min < max <= 1, "
                f"got {self.focus_min}/{self.focus_max}"
            )

    def is_conditioned(self, base_sigma: float) -> bool:
        return self.sigma_min <= base_sigma <= self.sigma_max

    def anchor_multiplier(self, lower: float = 0.0, upper: float = 1.0) -> float:
        return self.preservation_weight * preservation_density_compensation(
            self.sigma_max,
            self.focus_min,
            self.focus_max,
            self.focus_prob,
            self.sigma_min,
            lower,
            upper,
        )

    def describe(self) -> str:
        return (
            f"teacher matching: conditions={self.conditions} caption={self.caption_style} "
            f"sigma=[{self.sigma_min}, {self.sigma_max}] dc_weight={self.dc_weight} "
            f"mag_weight={self.mag_weight} preservation_weight={self.preservation_weight} "
            f"bleed={self.bleed_strength} focus=(p={self.focus_prob}, "
            f"[{self.focus_min}, {self.focus_max}])"
        )


# ---------------------------------------------------------------------------
# Teacher caption wraps
# ---------------------------------------------------------------------------

# The ref-teacher caption wrap: the official editing-prompt declaration blocks
# that make the base treat the reference as a 1:1 copy source. The
# `<Audio 1>: fully_copy` declaration is what opens audio education across the
# teaching band. Wording from musubi-tuner (Apache-2.0).
REF_TEACHER_CAPTION_HEADER_VIDEO = """subject_definitions:
<Video 1> is the source video for the target video edit.
<Audio 1> is the synchronized audio track of <Video 1> and is reused in the target video.

summary:
[video editing + audio reuse] The target video is an edited version of <Video 1> with no changes; all shots, subjects, camera movement, and sound are preserved as they are.

retention_analysis:
<Video 1> (all shots): fully_preserved - every shot, subject, action, and camera movement of the source video is retained without modification.
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.

detailed_description:
"""

# musubi's ref teacher is video-only; this is the same declaration shape for a
# still image reference (image datasets train as single latent frames)
REF_TEACHER_CAPTION_HEADER_IMAGE = """subject_definitions:
<Picture 1> is the source image for the target image edit.

summary:
[image editing] The target image is an edited version of <Picture 1> with no changes; all subjects, composition, and framing are preserved as they are.

retention_analysis:
<Picture 1>: fully_preserved - every subject, composition, and detail of the source image is retained without modification.

detailed_description:
"""


def wrap_ref_teacher_caption(caption: str, *, is_video: bool) -> str:
    header = REF_TEACHER_CAPTION_HEADER_VIDEO if is_video else REF_TEACHER_CAPTION_HEADER_IMAGE
    return header + caption


# The subject-reference teacher caption wrap: the official full-reference
# declaration blocks that make the base read each picture as a *subject*
# (identity/appearance) reference rather than a frame of the target.
# `attribute_transfer` is the empirically validated marker and the "pose,
# framing, outfit and setting follow the description" clause is what keeps
# the picture from acting as a copy source. Wording from musubi-tuner
# (Apache-2.0).
SUBJECT_REF_SUMMARY_IMAGE = (
    "The target is a single still image with no motion, a static shot of {subjects} as described below."
)
SUBJECT_REF_SUMMARY_VIDEO = "The target video shows {subjects} as described below."


def wrap_subject_reference_caption(
    caption: str, image_count: int, *, still_image: bool, ref_label: str = "Picture"
) -> str:
    if image_count < 1:
        raise ValueError("subject-reference teacher caption requires at least one picture")
    labels = [f"<Subject {i}>" for i in range(1, image_count + 1)]
    if len(labels) == 1:
        subjects = labels[0]
    else:
        subjects = ", ".join(labels[:-1]) + f" and {labels[-1]}"
    definitions = "\n".join(
        f"<Subject {i}> is the subject whose appearance comes from <{ref_label} {i}> (face and hair style)."
        for i in range(1, image_count + 1)
    )
    summary = (SUBJECT_REF_SUMMARY_IMAGE if still_image else SUBJECT_REF_SUMMARY_VIDEO).format(
        subjects=subjects
    )
    retention = "\n".join(
        f"<Subject {i}> (appears in [Shot 1]): attribute_transfer - the appearance of <Subject {i}> in"
        f" <{ref_label} {i}> is referenced; pose, framing, outfit and setting follow the description."
        for i in range(1, image_count + 1)
    )
    return (
        f"subject_definitions:\n{definitions}\n\n"
        f"summary:\n[reference generation] {summary}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{caption}"
    )
