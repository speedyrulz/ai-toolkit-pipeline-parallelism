"""Calibrated identity scoring math (docs/identity_eval_plan.md, spec §5-§6).

Pure python with no third-party imports on purpose: both the trainer and the
evaluator worker import it, and the worker runs in its own environment. The
worker imports this file directly by path, so keep it free of toolkit imports.

Per feature j and eval slot s:
    B_js = raw_j of slot s at step 0                    (paired baseline)
    T_j  = quantile_q of leave-one-out reference raws    (target)
    score_js = 100 * clip((raw - B_js) / (T_j - B_js), 0, 1)
Feature scores combine into a modality score with fixed weights (complete
case only), slots aggregate per pose bin (median) then macro-average across
bins, and face and body fuse with a weighted geometric mean.
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

EPS = 1e-6

# score status values (spec §8.3)
VALID = "VALID"
NO_FACE = "NO_FACE"
BODY_CROPPED = "BODY_CROPPED"
UNCALIBRATED = "UNCALIBRATED"
MISSING_MODALITY = "MISSING_MODALITY"
FAILED = "FAILED"

# minimum share of valid slots and distinct pose bins to publish a headline
MIN_COVERAGE = 0.8
MIN_BINS = 2


def clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def quantile(values: Iterable[float], q: float) -> Optional[float]:
    """Linear-interpolated quantile (numpy's default method); None if empty."""
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def median(values: Iterable[float]) -> Optional[float]:
    return quantile(values, 0.5)


def target(loo_raws: Iterable[float], q: float = 0.75, min_pairs: int = 2) -> Optional[float]:
    """100-point anchor from leave-one-out reference comparisons."""
    vals = [v for v in loo_raws if v is not None and math.isfinite(v)]
    if len(vals) < min_pairs:
        return None
    return quantile(vals, q)


def feature_score(
    raw: Optional[float], baseline: Optional[float], tgt: Optional[float]
) -> Tuple[Optional[float], Optional[float], str]:
    """(score 0-100, unclipped progress ratio, status) for one feature of one
    slot. The ratio is kept for the memorization warning (>1 = past target)."""
    if raw is None or not math.isfinite(raw):
        return None, None, FAILED
    if baseline is None or tgt is None or tgt <= baseline + EPS:
        return None, None, UNCALIBRATED
    ratio = (raw - baseline) / (tgt - baseline)
    return 100.0 * clip(ratio, 0.0, 1.0), ratio, VALID


def weighted_score(scores: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    """Fixed-weight combination, complete case: any weighted feature missing
    gives None rather than silently renormalizing the others."""
    total_w = sum(w for w in weights.values() if w > 0)
    if total_w <= 0:
        return None
    acc = 0.0
    for name, w in weights.items():
        if w <= 0:
            continue
        s = scores.get(name)
        if s is None:
            return None
        acc += w * s
    return clip(acc / total_w)


def aggregate(slots: List[dict], expected_slots: int) -> dict:
    """Checkpoint-level modality score from per-slot results.

    slots: [{'bin': str, 'score': float | None}] for every slot of the suite.
    Median per pose bin, then the unweighted mean over bins. Published only
    when coverage and bin count meet the gates; otherwise score is None with
    MISSING_MODALITY.
    """
    valid = [s for s in slots if s.get('score') is not None]
    coverage = (len(valid) / expected_slots) if expected_slots > 0 else 0.0
    by_bin: Dict[str, List[float]] = {}
    for s in valid:
        by_bin.setdefault(s.get('bin') or 'any', []).append(s['score'])
    bin_scores = {b: median(v) for b, v in by_bin.items()}
    out = {
        'score': None,
        'coverage': coverage,
        'by_bin': bin_scores,
        'valid_slots': len(valid),
        'status': MISSING_MODALITY,
    }
    if bin_scores and coverage + EPS >= MIN_COVERAGE and len(bin_scores) >= min(MIN_BINS, expected_slots):
        out['score'] = clip(sum(bin_scores.values()) / len(bin_scores))
        out['status'] = VALID
    return out


def fuse(face: Optional[float], body: Optional[float], w_face: float = 0.55, w_body: float = 0.45) -> Optional[float]:
    """Weighted geometric mean: a zero in either modality forces zero, 100
    needs both at 100, a missing modality gives None (never a substituted 0)."""
    if face is None or body is None:
        return None
    total = w_face + w_body
    wf, wb = w_face / total, w_body / total
    return clip(100.0 * (clip(face) / 100.0) ** wf * (clip(body) / 100.0) ** wb)


def score_slot(
    raws: Dict[str, Optional[float]],
    baselines: Dict[str, Optional[float]],
    targets: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> dict:
    """Calibrated scores for one generated image (one slot)."""
    features = {}
    for name in weights:
        score, ratio, status = feature_score(raws.get(name), baselines.get(name), targets.get(name))
        features[name] = {'raw': raws.get(name), 'score': score, 'ratio': ratio, 'status': status}
    combined = weighted_score({k: v['score'] for k, v in features.items()}, weights)
    max_ratio = max((v['ratio'] for v in features.values() if v['ratio'] is not None), default=None)
    return {'score': combined, 'features': features, 'max_ratio': max_ratio}
