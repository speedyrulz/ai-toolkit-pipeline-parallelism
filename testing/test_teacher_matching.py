"""CPU unit tests for toolkit/teacher_matching.py (no model weights needed).

Run: python -m pytest testing/test_teacher_matching.py
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toolkit.teacher_matching import (  # noqa: E402
    TEACHER_CONDITIONS_REF,
    TEACHER_CONDITIONS_SUBJECT_REF,
    TeacherMatchingConfig,
    base_sigma_from_uniform,
    dc_attenuated_prediction,
    decomposed_flow_loss,
    normalize_teacher_conditions,
    prediction_geometry_log,
    preservation_density_compensation,
    shift_sigma,
    unshift_sigma,
    wrap_ref_teacher_caption,
    wrap_subject_reference_caption,
)


# --- sigma helpers ---------------------------------------------------------


def test_unshift_inverts_shift():
    base = torch.linspace(0.0, 1.0, 11)
    for shift in (3.0, 12.0, 10.0):
        assert torch.allclose(unshift_sigma(shift_sigma(base, shift), shift), base, atol=1e-6)


def test_timestep_focus_remaps_a_uniform_draw_into_the_band_mixture():
    u = torch.tensor([0.0, 0.25, 0.4999, 0.5, 0.75, 1.0])
    base = base_sigma_from_uniform(u, focus_min=0.4, focus_max=0.8, focus_prob=0.5)
    # the first half of the draw sweeps the focus band
    assert torch.allclose(base[:3], torch.tensor([0.4, 0.6, 0.79992]), atol=1e-4)
    # the second half sweeps the full range
    assert torch.allclose(base[3:], torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)


def test_timestep_focus_composes_with_the_clipped_base_range():
    u = torch.tensor([0.0, 0.5, 1.0])
    base = base_sigma_from_uniform(u, lower=0.2, upper=0.6, focus_prob=0.0)
    assert torch.allclose(base, torch.tensor([0.2, 0.4, 0.6]))
    base = base_sigma_from_uniform(
        u, lower=0.2, upper=0.6, focus_min=0.3, focus_max=0.5, focus_prob=0.5
    )
    assert torch.allclose(base, torch.tensor([0.3, 0.2, 0.6]))


def test_focus_mixture_has_the_expected_band_density():
    torch.manual_seed(0)
    u = torch.rand(200_000)
    base = base_sigma_from_uniform(u, focus_min=0.4, focus_max=0.8, focus_prob=0.5)
    in_band = ((base >= 0.4) & (base < 0.8)).float().mean().item()
    # p + (1 - p) * band width = 0.5 + 0.5 * 0.4
    assert abs(in_band - 0.7) < 0.01


# --- loss shaping ----------------------------------------------------------


def test_dc_attenuation_scales_only_the_residual_dc_component():
    torch.manual_seed(0)
    target = torch.randn(2, 4, 3, 5, 5)
    pred = target + torch.randn(2, 4, 3, 5, 5) * 0.1 + 0.7  # a global cast on top of noise
    residual = pred - target
    dc = residual.mean(dim=(2, 3, 4), keepdim=True)
    ac = residual - dc
    mse_dc = dc.pow(2).mean()
    mse_ac = ac.pow(2).mean()
    for w in (1.0, 0.3, 0.0):
        attenuated = dc_attenuated_prediction(pred, target, w)
        mse = (attenuated - target).pow(2).mean()
        assert torch.allclose(mse, mse_ac + w * mse_dc, atol=1e-5)


def test_dc_attenuation_leaves_gradients_exact():
    torch.manual_seed(0)
    target = torch.randn(1, 2, 4, 4)
    pred = (target + 0.5).requires_grad_(True)
    attenuated = dc_attenuated_prediction(pred, target, 0.3)
    (attenuated - target).pow(2).mean().backward()
    assert torch.isfinite(pred.grad).all()


def test_decomposed_loss_equals_per_sample_mse_at_unit_weights():
    torch.manual_seed(0)
    pred = torch.randn(3, 4, 2, 6, 6)
    target = torch.randn(3, 4, 2, 6, 6)
    loss = decomposed_flow_loss(pred, target, 1.0, 1.0)
    assert loss.shape == (3,)
    mse = (pred - target).pow(2).flatten(1).mean(dim=1)
    assert torch.allclose(loss, mse, atol=1e-5)


def test_decomposed_loss_direction_gradient_is_rotational():
    """With the magnitude term off, the gradient of the direction term is
    orthogonal to the prediction (it cannot shrink the norm)."""
    torch.manual_seed(0)
    pred = torch.randn(1, 8, 4, 4).requires_grad_(True)
    target = torch.randn(1, 8, 4, 4)
    decomposed_flow_loss(pred, target, mag_weight=0.0, dir_weight=1.0).sum().backward()
    dot = (pred.grad.flatten() @ pred.detach().flatten()).item()
    assert abs(dot) < 1e-4 * pred.grad.norm().item() * pred.norm().item()


def test_decomposed_loss_magnitude_term_pulls_the_norm_to_the_target():
    pred = torch.ones(1, 2, 2, 2) * 2.0
    target = torch.ones(1, 2, 2, 2)  # same direction, half the norm
    loss = decomposed_flow_loss(pred, target, mag_weight=1.0, dir_weight=1.0)
    n = pred.numel()
    expected = (pred.norm() - target.norm()).pow(2) / n
    assert torch.allclose(loss, expected.view(1), atol=1e-6)
    assert decomposed_flow_loss(pred, target, mag_weight=0.0).abs().item() < 1e-6


def test_preservation_density_compensation_restores_the_anchor_share_under_focus():
    # anchor band (0.75, 1.0] is 25% of a uniform draw; a 0.5 focus on
    # [0.4, 0.8] leaves it (1 - 0.5) * 0.25 + 0.5 * (0.05 / 0.4) = 0.1875
    m = preservation_density_compensation(0.75, 0.4, 0.8, 0.5)
    assert math.isclose(m, 0.25 / 0.1875, rel_tol=1e-9)
    # no focus, or no anchor band: no correction
    assert preservation_density_compensation(0.75, 0.4, 0.8, 0.0) == 1.0
    assert preservation_density_compensation(1.0, 0.4, 0.8, 0.5) == 1.0


def test_preservation_density_compensation_counts_the_lower_anchor_band():
    # sigma_min 0.15 adds [0, 0.15) to the anchor; the focus band never
    # overlaps it: (0.25 + 0.15) / ((1 - 0.5) * 0.4 + 0.5 * 0.125) = 0.4 / 0.2625
    m = preservation_density_compensation(0.75, 0.4, 0.8, 0.5, sigma_min=0.15)
    assert math.isclose(m, 0.4 / 0.2625, rel_tol=1e-9)


def test_preservation_density_compensation_measures_inside_the_clipped_range():
    # range [0, 0.9]: anchor is (0.75, 0.9] = 0.15 of 0.9
    m = preservation_density_compensation(0.75, 0.4, 0.8, 0.5, lower=0.0, upper=0.9)
    uniform = 0.15 / 0.9
    focused = 0.5 * uniform + 0.5 * (0.05 / 0.4)
    assert math.isclose(m, uniform / focused, rel_tol=1e-9)


def test_prediction_geometry_log_splits_the_residual_into_dc_and_ac():
    torch.manual_seed(0)
    target = torch.randn(2, 3, 4, 4)
    pred = target + 0.5  # pure DC offset
    logs = prediction_geometry_log("video", pred, target)
    assert logs["teacher/video_residual_ac_rms"] < 1e-6
    assert math.isclose(logs["teacher/video_residual_dc_rms"], 0.5, rel_tol=1e-5)
    same = prediction_geometry_log("video", target, target)
    assert math.isclose(same["teacher/video_cos"], 1.0, abs_tol=1e-6)
    assert math.isclose(same["teacher/video_norm_ratio"], 1.0, abs_tol=1e-6)


# --- config ----------------------------------------------------------------


def test_config_defaults_leave_teacher_matching_off():
    cfg = TeacherMatchingConfig({})
    assert not cfg.enabled
    assert cfg.conditions is None


def test_config_legacy_dopsd_spelling_maps_to_the_ref_teacher_with_token_captions():
    cfg = TeacherMatchingConfig({"dopsd": True, "dopsd_bleed_strength": 0.5})
    assert cfg.enabled
    assert cfg.conditions == TEACHER_CONDITIONS_REF
    assert cfg.caption_style == "token"
    assert cfg.bleed_strength == 0.5
    assert cfg.sigma_max == 1.0 and cfg.focus_prob == 0.0


def test_config_musubi_recipe():
    cfg = TeacherMatchingConfig(
        {
            "teacher_matching": True,
            "teacher_conditions": "subject_ref",
            "teacher_condition_sigma_min": 0.15,
            "teacher_loss_mag_weight": 0.5,
            "teacher_loss_dc_weight": 0.3,
            "timestep_focus_prob": 0.5,
        }
    )
    assert cfg.conditions == TEACHER_CONDITIONS_SUBJECT_REF
    assert cfg.caption_style == "musubi"
    assert cfg.bleed_strength == 0.0
    assert cfg.is_conditioned(0.5) and not cfg.is_conditioned(0.1)
    assert cfg.anchor_multiplier() == pytest.approx(
        preservation_density_compensation(1.0, 0.4, 0.8, 0.5, sigma_min=0.15)
    )


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"teacher_matching": True, "teacher_conditions": "first,last"}, "teacher_conditions"),
        ({"teacher_matching": True, "teacher_condition_sigma_max": 1.5}, "sigma_max"),
        ({"teacher_matching": True, "teacher_condition_sigma_min": 0.9, "teacher_condition_sigma_max": 0.5}, "sigma_min"),
        ({"teacher_matching": True, "timestep_focus_prob": 0.5, "timestep_focus_min": 0.8, "timestep_focus_max": 0.4}, "focus_min"),
        ({"teacher_matching": True, "teacher_conditions": "subject_ref", "teacher_caption_style": "token"}, "token"),
    ],
)
def test_config_rejects_invalid_values(kw, message):
    with pytest.raises(ValueError, match=message):
        TeacherMatchingConfig(kw)


def test_normalize_teacher_conditions_strips_whitespace():
    assert normalize_teacher_conditions(" ref ") == TEACHER_CONDITIONS_REF
    assert normalize_teacher_conditions("subject_ref") == TEACHER_CONDITIONS_SUBJECT_REF


# --- caption wraps ---------------------------------------------------------


def test_ref_teacher_caption_declares_the_clip_as_a_copy_source():
    wrapped = wrap_ref_teacher_caption("a cat", is_video=True)
    assert wrapped.endswith("detailed_description:\na cat")
    assert "<Audio 1>: fully_copy" in wrapped
    still = wrap_ref_teacher_caption("a cat", is_video=False)
    assert "<Picture 1>" in still and "<Audio" not in still


def test_subject_reference_caption_lists_every_picture():
    wrapped = wrap_subject_reference_caption("a cat", 2, still_image=False)
    assert "<Subject 1>, and <Subject 2>" not in wrapped
    assert "<Subject 1> and <Subject 2>" in wrapped
    assert wrapped.count("attribute_transfer") == 2
    assert "<Picture 2>" in wrapped
    assert wrapped.endswith("detailed_description:\na cat")
    still = wrap_subject_reference_caption("a cat", 1, still_image=True)
    assert "single still image" in still
    with pytest.raises(ValueError):
        wrap_subject_reference_caption("a cat", 0, still_image=True)


# --- trainer integration pattern --------------------------------------------


def test_per_sample_loss_survives_the_elementwise_reduction_pipeline():
    """SDTrainer expands the (B,) teacher loss to the prediction's shape so
    the shared mask/mean pipeline reduces it back unchanged, with gradients."""
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 3, 5, 5).requires_grad_(True)
    target = torch.randn(2, 4, 3, 5, 5)
    loss_b = decomposed_flow_loss(pred, target, 0.5, 1.0)
    expanded = loss_b.view(2, 1, 1, 1, 1).expand_as(pred)
    reduced = (expanded * torch.ones_like(pred)).mean([1, 2, 3, 4])
    assert torch.allclose(reduced, loss_b)
    reduced.mean().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
