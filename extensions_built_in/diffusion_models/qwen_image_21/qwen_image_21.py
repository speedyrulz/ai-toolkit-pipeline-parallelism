"""Qwen-Image-2.1 for ai-toolkit.

Qwen-Image-2.1 is a single-stream DiT text-to-image model (7B):
  - text encoder: Qwen3-VL-8B (last decoder layer's hidden states, pre-RMSNorm),
  - autoencoder: a 16x-spatial (f16) video-style VAE with 64 latent channels and
    RGBA (4-channel) pixels, latents normalized by config mean/std,
  - denoiser: ``QwenImage21Transformer2DModel`` (vendored from diffusers main in
    ``src/``): 32 single-stream blocks, block-causal attention, one shared
    modulation projection, latents consumed unpatched (patch_size 1, one token
    per 16x16 pixel tile).

Flow-matching convention matches ai-toolkit (t=1 noise -> t=0 clean, target =
noise - clean). Text-to-image training only for now; the edit path (condition
images in the sequence) is inference-capable via the vendored pipeline but not
wired into training.
"""

import os
from typing import List, Optional

import torch
from transformers import AutoProcessor

from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.models.v2._mixin import OstrisModelMixin
from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21 as _AutoencoderKLQwenImage21,
)
from .src.pipeline_qwenimage21 import QwenImage21Pipeline
from .src.transformer_qwenimage21 import (
    QwenImage21Transformer2DModel as _QwenImage21Transformer2DModel,
)

# The reference repo; supplies configs (and weights when loading by repo id).
BASE_REPO = "Qwen/Qwen-Image-2.1"

HF_TOKEN = os.getenv("HF_TOKEN", None)

# mirrors the repo's scheduler/scheduler_config.json
scheduler_config = {
    "base_image_seq_len": 256,
    "max_image_seq_len": 8192,
    "base_shift": 0.5,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
}


class QwenImage21Transformer(_QwenImage21Transformer2DModel, OstrisModelMixin):
    aitk_subfolder = "transformer"
    aitk_config_repo = BASE_REPO

    aitk_comfy_repo = "Comfy-Org/Qwen-Image-2.1"
    # the comfy files use the diffusers key layout directly
    aitk_comfy_weight_names = {
        BASE_REPO: [
            "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
            "diffusion_models/qwen_image_2.1_bf16.safetensors",
        ],
    }

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks"]

    @classmethod
    def get_quantization_exclude_modules(cls):
        # sensitive modules kept in full precision (fnmatch patterns):
        #   img_in / txt_in*     - latent and text input projections
        #   time_text_embed*     - timestep embedder
        #   modulation*          - the single shared modulation feeding every block
        #   norm_out* / proj_out - final adaLN + output projection
        return [
            "img_in",
            "txt_in*",
            "time_text_embed*",
            "modulation*",
            "norm_out*",
            "proj_out",
        ]

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        # tolerate single-file checkpoints that carry a comfy module prefix
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if any(k.startswith(prefix) for k in state_dict):
                state_dict = {
                    (k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in state_dict.items()
                }
                break
        return state_dict


class QwenImage21VAE(_AutoencoderKLQwenImage21, OstrisModelMixin):
    aitk_subfolder = "vae"
    aitk_config_repo = BASE_REPO

    aitk_comfy_repo = "Comfy-Org/Qwen-Image-2.1"
    aitk_comfy_weight_names = {
        BASE_REPO: [
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ],
    }


class QwenImage21Model(BaseModel):
    arch = "qwen_image_21"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["QwenImage21Transformer2DModel"]
        self.processor = None

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # f16 VAE; the transformer additionally groups target tokens 2x2 into
        # the vision-language image slots, so latent h/w must be even
        return 16 * 2

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen-Image-2.1 model")

        model_path = self.model_config.name_or_path
        # a single-file checkpoint pulls its config and the other components
        # from the base repo
        extras_path = self.model_config.extras_name_or_path
        if extras_path is None or extras_path.endswith(".safetensors"):
            extras_path = BASE_REPO
        if model_path.endswith(".safetensors") or os.path.isfile(model_path):
            components_path = extras_path
        else:
            components_path = model_path

        self.print_and_status_update("Loading transformer")
        transformer = QwenImage21Transformer.load_model(
            model_path,
            dtype=dtype,
            token=HF_TOKEN,
            use_comfy_weights=self.model_config.model_kwargs.get(
                "use_comfy_weights", True
            ),
        )
        # quantize + offload + placement (incl. pipeline sharding), all driven
        # by model_config
        transformer.aitk_post_load(**self.component_load_kwargs("transformer"))
        flush()

        te_path = self.model_config.model_kwargs.get(
            "text_encoder_path", components_path
        )
        te_subfolder = "" if te_path != components_path else None
        if te_path == components_path:
            self.print_and_status_update(
                f"Loading Qwen3-VL-8B text encoder from {te_path}"
            )
            text_encoder = Qwen3VLTextEncoder.load_model(
                te_path, dtype=dtype, token=HF_TOKEN
            )
            processor = AutoProcessor.from_pretrained(
                components_path, subfolder="processor", token=HF_TOKEN
            )
        else:
            # a raw Qwen3-VL repo: weights and processor at the root
            self.print_and_status_update(
                f"Loading Qwen3-VL-8B text encoder from {te_path}"
            )
            text_encoder = Qwen3VLTextEncoder.load_model(
                te_path, dtype=dtype, subfolder="", token=HF_TOKEN
            )
            processor = AutoProcessor.from_pretrained(te_path, token=HF_TOKEN)
        # t2i training only encodes text; the vision tower is dead weight
        text_encoder.drop_vision_tower()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))
        flush()

        self.print_and_status_update("Loading Qwen-Image-2.1 VAE")
        vae_path = self.model_config.model_kwargs.get("vae_path", components_path)
        vae = QwenImage21VAE.load_model(
            vae_path, dtype=self.vae_torch_dtype, token=HF_TOKEN
        )
        vae.eval()
        vae.requires_grad_(False)
        vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)

        self.noise_scheduler = QwenImage21Model.get_train_scheduler()

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = processor.tokenizer
        self.processor = processor
        self.model = transformer
        self.pipeline = QwenImage21Pipeline(
            scheduler=self.noise_scheduler,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            transformer=transformer,
        )
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------
    def get_prompt_embeds(self, prompt) -> PromptEmbeds:
        # the te may be pinned to another gpu (te_device); encode there and
        # hand the embeddings back on the trainer's device
        te = unwrap_model(self.text_encoder)
        te_device = torch.device(self.te_device_torch)
        if te.device != te_device:
            te.to(te_device)

        prompt_embeds, prompt_embeds_mask, _ = self.pipeline.encode_prompt(
            prompt,
            device=te_device,
        )
        prompt_embeds = prompt_embeds.to(self.device_torch)
        # the pipeline returns None when nothing is padded
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(
                prompt_embeds.shape[:2],
                device=prompt_embeds.device,
                dtype=torch.int64,
            )
        else:
            prompt_embeds_mask = prompt_embeds_mask.to(self.device_torch)
        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = prompt_embeds_mask
        return pe

    # ------------------------------------------------------------------
    # VAE (RGBA, f16, video-style with a single frame, mean/std normalized)
    # ------------------------------------------------------------------
    def encode_images(self, image_list, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)
        # RGB [-1, 1] -> RGBA: fully opaque alpha is 1.0 in this scale
        if images.shape[1] == 3:
            images = torch.cat(
                [images, torch.ones_like(images[:, :1])], dim=1
            )
        images = images.unsqueeze(2)  # single frame dim
        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (latents - latents_mean) / latents_std
        latents = latents.squeeze(2)
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)
        latents = latents.unsqueeze(2)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        images = self.vae.decode(latents).sample
        images = images.squeeze(2)
        # RGBA [-1, 1] -> RGB composited over white
        if images.shape[1] == 4:
            rgb, alpha = images[:, :3], images[:, 3:]
            alpha01 = ((alpha + 1.0) / 2.0).clamp(0.0, 1.0)
            images = rgb * alpha01 + (1.0 - alpha01)
        return images.to(device, dtype=dtype)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 64, h, w)
        timestep: torch.Tensor,  # 0..1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)
        batch_size, num_channels_latents, height, width = latent_model_input.shape

        # 2.1 consumes latents unpatched: plain spatial flatten
        packed = latent_model_input.view(
            batch_size, num_channels_latents, height * width
        ).transpose(1, 2)
        img_shapes = [[(1, height, width)]] * batch_size

        enc_hs = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        prompt_embeds_mask = text_embeddings.attention_mask.to(
            self.device_torch, dtype=torch.int64
        )
        # text positions hold no image slots for t2i; one slot per 2x2 group of
        # target latents is appended
        img_mask = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    enc_hs.shape[1],
                    dtype=torch.bool,
                    device=self.device_torch,
                ),
                torch.ones(
                    batch_size,
                    (height * width) // 4,
                    dtype=torch.bool,
                    device=self.device_torch,
                ),
            ],
            dim=1,
        )

        noise_pred = self.transformer(
            hidden_states=packed.to(self.device_torch, self.torch_dtype).detach(),
            timestep=(timestep / 1000).detach(),
            encoder_hidden_states=enc_hs.detach(),
            encoder_hidden_states_mask=prompt_embeds_mask.detach(),
            img_shapes=img_shapes,
            img_mask=img_mask,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, -packed.shape[1]:]

        noise_pred = noise_pred.transpose(1, 2).view(
            batch_size, num_channels_latents, height, width
        )
        return noise_pred

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    # ------------------------------------------------------------------
    # Sampling (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        pipeline = QwenImage21Pipeline(
            scheduler=QwenImage21Model.get_train_scheduler(),
            vae=unwrap_model(self.vae),
            text_encoder=unwrap_model(self.text_encoder),
            processor=self.processor,
            transformer=unwrap_model(self.transformer),
        )
        return pipeline

    def generate_single_image(
        self,
        pipeline: QwenImage21Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, self.torch_dtype
            ),
            prompt_embeds_mask=conditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.int64
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, self.torch_dtype
            ),
            negative_prompt_embeds_mask=unconditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.int64
            ),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            true_cfg_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]
        return img

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: QwenImage21Transformer = unwrap_model(self.model)
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
        )

    def get_base_model_version(self):
        return "qwen_image_2.1"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        # comfy/diffusers convention for this family
        return {
            k.replace("transformer.", "diffusion_model."): v
            for k, v in state_dict.items()
        }

    def convert_lora_weights_before_load(self, state_dict):
        return {
            k.replace("diffusion_model.", "transformer."): v
            for k, v in state_dict.items()
        }
