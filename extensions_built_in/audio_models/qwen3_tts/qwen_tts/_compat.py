# ai-toolkit: shims so the qwen_tts 0.1.1 sources (written against
# transformers 4.57) also run on transformers 5, where the rope helpers and
# the output-capture decorator were reworked. Behaviour matches 4.57.
import functools
from typing import Optional

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import generic as _generic


def _default_rope_parameters(config, device: Optional[torch.device] = None, seq_len: Optional[int] = None):
    """transformers 4.57 ``_compute_default_rope_parameters`` (v5 dropped the
    'default' entry and reads ``config.rope_parameters`` instead)."""
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def rope_init_fn(rope_type: str):
    if rope_type == "default" or rope_type not in ROPE_INIT_FUNCTIONS:
        if rope_type != "default":
            raise ValueError(f"qwen_tts: unsupported rope_type {rope_type!r}")
        return _default_rope_parameters
    return ROPE_INIT_FUNCTIONS[rope_type]


def check_model_inputs():
    """``@check_model_inputs()`` on 4.57; on 5.x the same job is split into
    ``merge_with_config_defaults`` + ``capture_outputs``."""
    merge = getattr(_generic, "merge_with_config_defaults", None)
    if merge is None:
        return _generic.check_model_inputs()
    from transformers.utils.output_capturing import capture_outputs

    def wrap(func):
        return functools.wraps(func)(merge(capture_outputs(func)))

    return wrap


def rope_config_validation(config):
    """No-op: it only validated rope_scaling, and v5's version requires
    ``config.rope_parameters``. The shipped configs all use default rope."""


# the pre-tokenizer split that transformers 4.57.3 installs for
# fix_mistral_regex=True (Qwen3-TTS configs say 4.57.3, which triggers it)
_FIXED_SPLIT_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?"
    r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def load_processor(path):
    """``AutoProcessor.from_pretrained(path, fix_mistral_regex=True)``. On
    transformers 5 that flag crashes (it patches a raw tokenizers.Tokenizer),
    so load without it and apply 4.57.3's exact pre-tokenizer change."""
    from transformers import AutoProcessor

    if getattr(_generic, "merge_with_config_defaults", None) is None:
        return AutoProcessor.from_pretrained(path, fix_mistral_regex=True)
    import tokenizers

    processor = AutoProcessor.from_pretrained(path)
    backend = processor.tokenizer.backend_tokenizer
    backend.pre_tokenizer[0] = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(_FIXED_SPLIT_REGEX), behavior="isolated")
    processor.tokenizer.fix_mistral_regex = True
    return processor


# _init_weights helpers: transformers 5 runs _init_weights over the whole
# model after loading and relies on transformers.initialization to skip
# tensors that were loaded; raw .data writes would wipe the checkpoint
try:
    from transformers import initialization as _hf_init
except ImportError:  # transformers 4.x
    _hf_init = None


def _loaded(tensor) -> bool:
    return bool(getattr(tensor, "_is_hf_initialized", False))


def init_normal_(tensor, std):
    if _hf_init is not None:
        return _hf_init.normal_(tensor, mean=0.0, std=std)
    return tensor.data.normal_(mean=0.0, std=std)


def init_zeros_(tensor):
    if _hf_init is not None:
        return _hf_init.zeros_(tensor)
    return tensor.data.zero_()


def init_ones_(tensor):
    if _hf_init is not None:
        return _hf_init.ones_(tensor)
    return tensor.data.fill_(1.0)


def init_padding_row_(embedding):
    if embedding.padding_idx is not None and not _loaded(embedding.weight):
        with torch.no_grad():
            embedding.weight[embedding.padding_idx].zero_()


@torch.no_grad()
def restore_rotary_buffers(model):
    """transformers 5 builds models on the meta device and only re-creates
    rotary buffers for modules its own init recognises; these models' rotary
    modules keep a zeroed inv_freq (every position looks the same). Recompute
    them the way __init__ does on 4.57."""
    for module in model.modules():
        fn = getattr(module, "rope_init_fn", None)
        buf = getattr(module, "inv_freq", None)
        if fn is None or not isinstance(buf, torch.Tensor):
            continue
        inv_freq, scaling = fn(module.config, buf.device)
        buf.copy_(inv_freq.to(buf.dtype))
        module.attention_scaling = scaling
        module.original_inv_freq = buf


def match_457_mimi_mask(encoder):
    """transformers 4.57 built the Mimi encoder's mask with create_causal_mask,
    so its 250-frame sliding window only applied under flash attention; 5.x
    applies the window everywhere. The 12 Hz codes Qwen's tools produce (and
    the ComfyUI pack, pinned to 4.57) come from the full causal mask, so the
    encoder keeps it: a window longer than any clip is plain causal."""
    if getattr(_generic, "merge_with_config_defaults", None) is None:
        return
    import copy

    transformer = getattr(encoder, "encoder_transformer", None)
    if transformer is None or getattr(transformer.config, "sliding_window", None) is None:
        return
    cfg = copy.copy(transformer.config)
    cfg.sliding_window = 1 << 30
    transformer.config = cfg


class CachePositionMixin:
    """transformers 5 stopped passing ``cache_position`` to forward during
    generate (except for remote-code models); the talker and code predictor
    use it to tell the prefill from decode steps. Recreate it the way 5.x's
    remote-code path does (what 4.57 passed)."""

    # explicit inputs_embeds: generate() inspects this signature for it
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      inputs_embeds=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, **kwargs)
        if model_inputs.get("cache_position") is None:
            ref = model_inputs.get("inputs_embeds")
            if ref is None:
                ref = model_inputs["input_ids"]
            past = model_inputs.get("past_key_values")
            seen = past.get_seq_length() if past is not None else 0
            model_inputs["cache_position"] = torch.arange(seen, seen + ref.shape[1], device=ref.device)
        return model_inputs
