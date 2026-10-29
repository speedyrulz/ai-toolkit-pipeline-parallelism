"""Fixed-seed evaluation prompt suites (spec §2.2).

Every slot keeps its prompt, seed, size and renderer for the whole run, step 0
included: only the network changes between rounds. [trigger] is replaced with
the job's trigger word like regular sample prompts.
"""

from typing import List, Optional, Union

DEFAULT_FACE_SUITE = [
    {"bin": "front", "prompt": "close-up portrait photo of [trigger], looking at the camera, neutral expression, plain grey background, soft even lighting"},
    {"bin": "three_quarter", "prompt": "close-up portrait photo of [trigger], head turned three-quarters to the left, plain grey background, soft even lighting"},
    {"bin": "three_quarter", "prompt": "close-up portrait photo of [trigger], head turned three-quarters to the right, plain grey background, soft even lighting"},
    {"bin": "profile", "prompt": "close-up side profile portrait photo of [trigger], facing left, plain grey background, soft even lighting"},
    {"bin": "front", "prompt": "close-up portrait photo of [trigger] smiling, looking at the camera, plain grey background"},
    {"bin": "front", "prompt": "close-up portrait photo of [trigger], looking at the camera, dramatic side lighting, dark background"},
]

DEFAULT_BODY_SUITE = [
    {"bin": "front", "prompt": "full body photo of [trigger] standing, front view, arms relaxed at sides, fitted clothing, plain white studio background"},
    {"bin": "three_quarter", "prompt": "full body photo of [trigger] standing, three-quarter view from the left, fitted clothing, plain white studio background"},
    {"bin": "three_quarter", "prompt": "full body photo of [trigger] standing, three-quarter view from the right, fitted clothing, plain white studio background"},
    {"bin": "side", "prompt": "full body photo of [trigger] standing, side view, fitted clothing, plain white studio background"},
    {"bin": "back", "prompt": "full body photo of [trigger] standing, back view, fitted clothing, plain white studio background"},
    {"bin": "front", "prompt": "full body photo of [trigger] sitting on a stool, front view, fitted clothing, plain white studio background"},
]

DEFAULT_FLEX_SUITE = [
    {"bin": "flex", "prompt": "photo of [trigger] wearing a red raincoat, walking on a city street at night, neon reflections"},
    {"bin": "flex", "prompt": "photo of [trigger] in a medieval knight costume, standing in a castle courtyard"},
    {"bin": "flex", "prompt": "photo of [trigger] reading a book in a cozy cafe, warm afternoon light"},
    {"bin": "flex", "prompt": "photo of [trigger] hiking on a mountain trail, wearing a backpack, wide shot"},
]

MODALITIES = ("face", "body", "flex")
RENDERERS = ("main", "helper")


def _parse_line(raw: str) -> dict:
    """'front | close-up portrait of [trigger]' -> bin + prompt; a line
    without ' | ' is a prompt in the 'any' bin."""
    if " | " in raw:
        bin_name, prompt = raw.split(" | ", 1)
        return {"bin": bin_name.strip(), "prompt": prompt.strip()}
    return {"prompt": raw.strip()}


def _normalize(items: List[Union[str, dict]], default: List[dict]) -> List[dict]:
    out = []
    for raw in (items or default):
        item = _parse_line(raw) if isinstance(raw, str) else dict(raw)
        if not item.get("prompt"):
            continue
        out.append(item)
    return out


def build_slots(cfg, sample_config, use_helper: bool) -> List[dict]:
    """Slot definitions for the three suites. cfg is an IdentityEvalConfig.
    Renderers alternate over the interleaved slot order so each card gets a
    mix of modalities; the assignment is saved in the manifest and reused."""
    base = sample_config.samples[0] if sample_config.samples else None
    width = base.width if base is not None else sample_config.width
    height = base.height if base is not None else sample_config.height
    steps = base.sample_steps if base is not None else sample_config.sample_steps
    guidance = base.guidance_scale if base is not None else sample_config.guidance_scale
    neg = (base.neg if base is not None else sample_config.neg) or ""
    seed0 = cfg.seed if cfg.seed is not None else int(sample_config.seed or 0)

    suites = {
        "face": _normalize(cfg.face_suite, DEFAULT_FACE_SUITE),
        "body": _normalize(cfg.body_suite, DEFAULT_BODY_SUITE),
        "flex": _normalize(cfg.flex_suite, DEFAULT_FLEX_SUITE),
    }
    slots = []
    for modality in MODALITIES:
        for i, item in enumerate(suites[modality]):
            slots.append({
                "id": f"{modality}_{i:02d}",
                "modality": modality,
                "bin": item.get("bin") or ("flex" if modality == "flex" else "any"),
                "prompt": item["prompt"],
                "neg": item.get("neg", neg) or "",
                "seed": int(item["seed"]) if item.get("seed") is not None else None,
                "width": int(item.get("width", width)),
                "height": int(item.get("height", height)),
                "steps": int(item.get("sample_steps", steps)),
                "guidance": float(item.get("guidance_scale", guidance)),
            })
    # seeds by global index so adding a flex prompt never reseeds face slots
    for n, s in enumerate(slots):
        if s["seed"] is None:
            s["seed"] = seed0 + n
    # interleave modalities, then alternate renderers
    order = sorted(range(len(slots)), key=lambda k: (int(slots[k]["id"][-2:]), MODALITIES.index(slots[k]["modality"])))
    for rank, k in enumerate(order):
        slots[k]["renderer"] = RENDERERS[rank % 2] if use_helper else "main"
    return slots


def slots_signature(slots: List[dict]) -> List[dict]:
    """The parts of each slot that the paired baseline depends on."""
    keys = ("id", "modality", "bin", "prompt", "neg", "seed", "width", "height", "steps", "guidance", "renderer")
    return [{k: s[k] for k in keys} for s in slots]
