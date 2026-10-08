"""OmniVoice (k2-fsa/OmniVoice) text-to-speech LoRA training (arch ``omnivoice``).

OmniVoice is a masked discrete-diffusion TTS model: a Qwen3 backbone reads
``[style tokens | text tokens | audio tokens]`` where the audio is 8 codebooks
of HiggsAudio-v2 tokens at 25 Hz, and predicts randomly masked audio tokens.
Training uses the omnivoice package's own sample processor (prompt/mask
ratios, condition dropout, language/instruct tokens) and the model's own
loss, routed through the trainer's ``is_llm`` path (no noise or scheduler).

Data: a folder of audio clips (wav/mp3/flac...) with a .txt transcript each.
The clip is tokenized once and cached (enable latent caching).

LoRA covers the q/k/v/o/gate/up/down projections of the 28 Qwen3 blocks (the
official recipe's set); the audio embeddings and heads stay frozen. Each save can
also be exported as a merged, standalone OmniVoice model folder into
ComfyUI's ``models/omnivoice`` folder, where the ComfyUI-OmniVoice-TTS nodes
list it like any other model (they cannot load LoRA files themselves).

Requires the ``omnivoice`` package (0.2.1, the version ComfyUI's
comfyui-omnivoice-tts installs). Install it WITHOUT its dependencies, because
it pins an old torch:  ``pip install omnivoice==0.2.1 --no-deps``

model_kwargs:
  language               language id for every clip, e.g. "en" (default: none)
  instruct               voice description for every clip, e.g.
                         "female, young adult, moderate pitch" (default: none)
  prompt_ratio_range     [lo, hi] share of a clip used as the voice prompt
  mask_ratio_range       [lo, hi] share of the remaining tokens masked
  drop_cond_ratio        chance a step drops text/prompt (keeps CFG working)
  language_ratio         chance the language token is shown
  instruct_ratio         chance the instruct text is shown
  only_instruct_ratio    chance an instruct step drops the audio prompt
  comfyui_export_dir     ComfyUI models/omnivoice folder to export merged
                         models into ("" = no export; a saved LoRA can be
                         exported later with scripts/export_omnivoice_lora.py).
                         \\\\wsl.localhost paths work.
  comfyui_export_every_save   export at every save (default true) or only
                         the final save
  comfyui_export_keep    exports of this job to keep (0 = all, default 0)
  comfyui_export_dtype   fp32 (default), fp16 or bf16. A LoRA's weight change
                         is small next to the base weights: merged into bf16,
                         about half of it rounds away (measured 49%; fp16 8%,
                         fp32 0%). Load exports in ComfyUI with dtype fp32.
  sample_language / sample_instruct   used for training samples
  sample_ref_audio + sample_ref_text  voice-clone the samples from this clip
"""


import os
import threading
from typing import List, Optional

import torch

from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dto import DTO
from toolkit.print import print_acc

from ..base_audio_model import BaseAudioModel
from .export import DTYPES, write_export

DEFAULT_MODEL = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000
# training samples are written at the trainer's audio rate
SAVE_SAMPLE_RATE = 48000


def _require_omnivoice():
    try:
        import omnivoice  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The omnivoice package is required for OmniVoice training. Install it without its "
            "dependencies (it pins an old torch): pip install omnivoice==0.2.1 --no-deps"
        ) from e


class OmniVoiceModel(BaseAudioModel):
    arch = "omnivoice"
    is_llm = True
    sample_rate = SAMPLE_RATE

    def __init__(self, device, model_config: ModelConfig, dtype="bf16", custom_pipeline=None,
                 noise_scheduler=None, **kwargs):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_transformer = True
        # matched from the root, then limited to the llm.layers blocks by the
        # network's transformer_only filter (get_transformer_block_names); the
        # audio tokenizer is detached from the module tree in load_model
        self.target_lora_modules = ["OmniVoice"]
        kw = self.model_config.model_kwargs
        self.language: Optional[str] = kw.get("language", None)
        self.instruct: Optional[str] = kw.get("instruct", None)
        self.prompt_ratio_range = tuple(kw.get("prompt_ratio_range", (0.0, 0.3)))
        self.mask_ratio_range = tuple(kw.get("mask_ratio_range", (0.0, 1.0)))
        self.drop_cond_ratio = float(kw.get("drop_cond_ratio", 0.1))
        self.language_ratio = float(kw.get("language_ratio", 0.8))
        self.instruct_ratio = float(kw.get("instruct_ratio", 1.0))
        self.only_instruct_ratio = float(kw.get("only_instruct_ratio", 0.5))
        self.export_dir: str = str(kw.get("comfyui_export_dir", "") or "")
        self.export_every_save = bool(kw.get("comfyui_export_every_save", True))
        self.export_keep = int(kw.get("comfyui_export_keep", 0))
        self.export_dtype = DTYPES[str(kw.get("comfyui_export_dtype", "fp32")).lower()]
        # one export writes in the background while training continues
        self._export_thread = None
        self.sample_language = kw.get("sample_language", self.language)
        self.sample_instruct = kw.get("sample_instruct", self.instruct)
        self.sample_ref_audio = kw.get("sample_ref_audio", None)
        self.sample_ref_text = kw.get("sample_ref_text", None)
        self.processor = None
        self.audio_tokenizer = None
        self.base_path: Optional[str] = None
        self.additional_loss_logs = {}

    @staticmethod
    def get_train_scheduler():
        # masked discrete diffusion: the model samples its own mask ratios
        return None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def load_model(self):
        _require_omnivoice()
        from omnivoice import OmniVoice
        from omnivoice.data.processor import OmniVoiceSampleProcessor
        from omnivoice.models.omnivoice import _resolve_model_path

        if self.model_config.quantize:
            print_acc("omnivoice: quantization is not supported for this small model; ignoring quantize")
        if self.model_config.layer_offloading:
            raise NotImplementedError("Layer offloading is not implemented for omnivoice")

        name_or_path = self.model_config.name_or_path or DEFAULT_MODEL
        self.print_and_status_update(f"Loading OmniVoice from {name_or_path}")
        self.base_path = _resolve_model_path(name_or_path)
        model = OmniVoice.from_pretrained(
            self.base_path, dtype=self.torch_dtype, device_map=str(self.device_torch),
            attn_implementation="sdpa",
        )
        # keep the HiggsAudio tokenizer out of the module tree: the LoRA network
        # must not wrap its layers and the exported state dict must not carry it
        tokenizer = model._modules.pop("audio_tokenizer", None)
        if tokenizer is not None:
            object.__setattr__(model, "audio_tokenizer", tokenizer)
            tokenizer.requires_grad_(False)
            tokenizer.eval()
        self.audio_tokenizer = model.audio_tokenizer
        model.requires_grad_(False)
        model.eval()

        def enable_gradient_checkpointing():
            model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        model.enable_gradient_checkpointing = enable_gradient_checkpointing

        cfg = model.config
        self.processor = OmniVoiceSampleProcessor(
            text_tokenizer=model.text_tokenizer,
            num_channels=cfg.num_audio_codebook,
            audio_mask_id=cfg.audio_mask_id,
            prompt_ratio_range=self.prompt_ratio_range,
            mask_ratio_range=self.mask_ratio_range,
            drop_cond_ratio=self.drop_cond_ratio,
            language_ratio=self.language_ratio,
            use_pinyin_ratio=0.0,
            instruct_ratio=self.instruct_ratio,
            only_instruct_ratio=self.only_instruct_ratio,
        )
        self.model = model
        self.tokenizer = model.text_tokenizer
        self.text_encoder = None
        self.vae = None
        self.pipeline = self
        self.print_and_status_update("Model Loaded")
        if not self.export_dir:
            print_acc("omnivoice: comfyui_export_dir is empty, so saves are LoRA files only and will not "
                      "appear in ComfyUI (its OmniVoice nodes cannot load LoRAs). Set comfyui_export_dir in "
                      "the model kwargs, or export a finished LoRA later with "
                      "scripts/export_omnivoice_lora.py")

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["llm.layers"]

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def get_bucket_divisibility(self):
        return 1

    def save_model(self, output_path, meta, save_dtype):
        raise NotImplementedError("omnivoice: only LoRA training is supported (no full-model save)")

    # ------------------------------------------------------------------
    # audio -> tokens (latent cache)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, image_list: torch.Tensor, device=None, dtype=None):
        """[1, C, S] waveform at 24 kHz -> DTO carrying [1, 8, T] int32 tokens.
        The tokens ride as an int extra: float casts in the latent cache would
        round codes above 256 in bf16."""
        if image_list.shape[0] != 1:
            raise ValueError("omnivoice encodes one clip at a time (enable latent caching, batch_size 1)")
        wav = image_list[0].float()
        wav = wav.mean(0) if wav.shape[0] > 1 else wav[0]
        fe = self.model.feature_extractor
        inputs = fe(raw_audio=wav.cpu().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        tok_dev = next(self.audio_tokenizer.parameters()).device
        codes = self.audio_tokenizer.encode(inputs["input_values"].to(tok_dev)).audio_codes.squeeze(0)
        codes = codes.to(torch.int32)  # [8, T]
        return DTO(codes.float()[None], audio_tokens=codes[None])

    def encode_audio(self, audio_data_list):
        return torch.zeros(len(audio_data_list), 1)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def get_llm_loss(self, batch) -> torch.Tensor:
        from omnivoice.data.collator import PaddingDataCollator

        lat = batch.latents
        if not isinstance(lat, DTO) or lat.get("audio_tokens") is None:
            if batch.tensor is None:
                raise ValueError("omnivoice got a batch with neither audio nor cached tokens")
            lat = self.encode_images(batch.tensor)
        tokens = lat.get("audio_tokens")
        captions = batch.get_caption_list()
        samples = []
        for i in range(tokens.shape[0]):
            label = {"text": captions[i].strip()}
            if self.language:
                label["language_id"] = self.language
            if self.instruct:
                label["instruct"] = self.instruct
            samples.append(self.processor({"audio_tokens": tokens[i].long().cpu(), "label": label}))
        collated = PaddingDataCollator(self.processor, batch_tokens=0)(samples)
        collated = {k: v.to(self.device_torch) for k, v in collated.items()}
        out = self.model(
            input_ids=collated["input_ids"], audio_mask=collated["audio_mask"], labels=collated["labels"],
            attention_mask=collated["attention_mask"], position_ids=collated["position_ids"],
        )
        loss = out.loss
        self.additional_loss_logs = {"loss/ce": loss.detach().float().item()}
        return loss

    # LoRA keys relative to the OmniVoice root (llm.layers.N...., audio_heads)
    def convert_lora_weights_before_save(self, state_dict):
        out = {}
        for k, v in state_dict.items():
            for prefix in ("transformer.", "diffusion_model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            out[k] = v
        return out

    def convert_lora_weights_before_load(self, state_dict):
        return {(k if k.startswith("transformer.") else "transformer." + k): v for k, v in state_dict.items()}

    # ------------------------------------------------------------------
    # samples: the prompt is the text to speak
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return self

    def generate_single_image(self, pipeline, gen_config: GenerateImageConfig, conditional_embeds,
                              unconditional_embeds, generator, extra):
        import numpy as np
        import torchaudio
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

        if gen_config.output_ext not in ("mp3", "wav"):
            gen_config.output_ext = "wav"
        text = (gen_config.prompt or "").strip() or "Hello, this is a test."
        steps = int(gen_config.num_inference_steps) if gen_config.num_inference_steps else 32
        gen_cfg = OmniVoiceGenerationConfig(
            num_step=max(4, min(64, steps)),
            guidance_scale=float(gen_config.guidance_scale) if gen_config.guidance_scale is not None else 2.0,
        )
        kwargs = dict(text=text, language=self.sample_language, generation_config=gen_cfg)
        ref_audio = gen_config.ctrl_img or self.sample_ref_audio
        if ref_audio and self.sample_ref_text:
            kwargs.update(ref_audio=ref_audio, ref_text=self.sample_ref_text)
        elif self.sample_instruct:
            kwargs["instruct"] = self.sample_instruct
        torch.manual_seed(int(gen_config.seed) if gen_config.seed is not None else 0)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            audio = self.model.generate(**kwargs)[0]
        if was_training:
            self.model.train()
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if wav.ndim == 1:
            wav = wav[None]
        wav = torchaudio.functional.resample(wav, SAMPLE_RATE, SAVE_SAMPLE_RATE)
        return wav[None]  # [1, C, S]; save_image writes image[0] at 48 kHz

    # ------------------------------------------------------------------
    # ComfyUI export: merged standalone model folder per save
    # ------------------------------------------------------------------
    def after_network_save(self, lora_path: str, network, is_final: bool):
        if not self.export_dir:
            return
        if not is_final and not self.export_every_save:
            return
        try:
            self._export_merged(lora_path, network, wait=is_final)
        except Exception as e:  # an export failure must never stop training
            print_acc(f"omnivoice: ComfyUI export failed: {type(e).__name__}: {e}")

    @torch.no_grad()
    def _merged_state_dict(self, network) -> dict:
        """Base weights with every LoRA delta folded in, computed on stand-in
        copies of the wrapped layers so the live (frozen) weights are never
        merged in and out (bf16 round trips would drift them)."""
        names = {id(m): n for n, m in self.model.named_modules()}
        dt = self.export_dtype
        merged = {k: (v.detach().to("cpu", dt) if v.is_floating_point() else v.detach().cpu()).clone()
                  for k, v in self.model.state_dict().items()}
        prev_mult = network.multiplier
        network.multiplier = 1.0
        network._update_torch_multiplier()
        try:
            for module in network.get_all_modules():
                org = module.org_module[0]
                name = names.get(id(org))
                if name is None:
                    continue
                stand_in = torch.nn.Linear(org.in_features, org.out_features, bias=org.bias is not None,
                                           device=org.weight.device, dtype=torch.float32)
                stand_in.weight.copy_(org.weight.float())
                if org.bias is not None:
                    stand_in.bias.copy_(org.bias.float())
                module.org_module[0] = stand_in
                prev_merged = getattr(module, "is_merged", None)
                try:
                    module.merge_in(1.0)
                finally:
                    module.org_module[0] = org
                    if prev_merged is not None:
                        module.is_merged = prev_merged
                merged[f"{name}.weight"] = stand_in.weight.detach().to("cpu", dt)
        finally:
            network.multiplier = prev_mult
            network._update_torch_multiplier()
        return merged

    def _export_merged(self, lora_path: str, network, wait: bool = False):
        # the merge reads the live LoRA weights, so it runs now (a few seconds);
        # writing the ~3 GB folder over a network share does not need to
        # block training
        state = self._merged_state_dict(network)
        self.wait_for_exports()
        thread = threading.Thread(target=self._write_export, args=(lora_path, state), daemon=False)
        thread.start()
        self._export_thread = thread
        if wait:
            self.wait_for_exports()

    def wait_for_exports(self):
        if self._export_thread is not None:
            self._export_thread.join()
            self._export_thread = None

    def _write_export(self, lora_path: str, state: dict):
        try:
            write_export(state, lora_path, self.base_path, self.export_dir, self.export_keep, print_acc)
        except Exception as e:
            print_acc(f"omnivoice: ComfyUI export failed: {type(e).__name__}: {e}")
