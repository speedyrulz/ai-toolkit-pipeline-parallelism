"""Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-*-Base) custom-voice LoRA training (arch ``qwen3_tts``).

Qwen3-TTS is an autoregressive TTS model: a "talker" (Qwen3 backbone) reads the
text and predicts the first of 16 codebooks of 12.5 Hz speech-tokenizer codes
per frame, and a small "code predictor" fills in the other 15. The voice comes
from a speaker embedding (the Base model's speaker encoder run on reference
audio). Fine-tuning teaches the model one voice and stores that embedding as a
named speaker, the way ComfyUI-Qwen3-TTS's Finetune node and Qwen's own
``finetuning/sft_12hz.py`` do. The trained model is used with the Custom
Voice node and that speaker name.

Training data: a folder of speech clips (wav/mp3/flac...) with a .txt
transcript each. Each clip is tokenized once and cached (enable latent
caching). The speaker embedding is the mean over the reference clips
(``ref_audio``; default: the clips of the dataset folder).

The objective follows the inference path. Qwen's sft_12hz.py (and the
ComfyUI pack, which copies it) feeds the text without the talker's
``text_projection`` that generation always applies, and shifts the labels
twice (by hand, then again in the causal-LM loss), training each frame to
predict the frame after next. Measured on the untrained 1.7B Base model with
real speech, the generation-consistent objective scores talker/sub-talker CE
4.78/5.97 against 8.81/9.10 (near chance) for the sft recipe.
``model_kwargs.recipe: official`` reproduces sft_12hz.py exactly.

The model code is a vendored copy of qwen-tts 0.1.1 (Apache-2.0, see
qwen_tts/LICENSE), patched to run on transformers 5 with results identical to
the original on transformers 4.57.3 (see qwen_tts/_compat.py).

model_kwargs:
  speaker_name           name of the trained voice in ComfyUI (default: the
                         job name). Type it as custom_speaker_name in the
                         Custom Voice node.
  ref_audio              file or folder whose clips give the speaker
                         embedding (default: the dataset folder)
  recipe                 fixed (default) or official (sft_12hz.py as is)
  sub_talker_weight      weight of the code-predictor loss (default 0.3)
  comfyui_export_dir     ComfyUI models/Qwen3-TTS folder to export merged
                         custom-voice models into ("" = no export; a saved
                         LoRA can be exported later with
                         scripts/export_qwen3_tts_lora.py).
                         \\\\wsl.localhost paths work.
  comfyui_export_every_save   export at every save (default true) or only
                         the final save
  comfyui_export_keep    exports of this job to keep (0 = all, default 0)
  comfyui_export_dtype   fp32 (default), fp16 or bf16. Load fp32 exports with
                         the Loader's precision set to fp32: merged into bf16,
                         much of a LoRA's small weight change rounds away.
  sample_language        language for training samples (default Auto)
  sample_instruct        optional style instruction for samples (1.7B only)
"""

import glob
import json
import os
import re
from typing import List, Optional

import torch
import torch.nn.functional as F

from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dto import DTO
from toolkit.print import print_acc

from ..base_audio_model import BaseAudioModel
from ..merged_export import DTYPES, merged_state_dict, write_export

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SAMPLE_RATE = 24000
SAVE_SAMPLE_RATE = 48000
FRAME_RATE = 12.5
# codec-embedding row that holds a custom voice (the ComfyUI pack and
# sft_12hz.py both use 3000; ids 2048+ are never generated as audio)
SPEAKER_ROW = 3000
# the speaker embedding rides in the LoRA file under this key, so a LoRA can
# be exported (or resumed) without the dataset
SPEAKER_KEY = "qwen3_tts.speaker_embedding"
SAMPLE_SPEAKER = "aitk_voice"
_AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac")
_MAX_REF_CLIPS = 32


def _speaker_name_from(lora_path: str) -> str:
    return re.sub(r"_\d+$", "", os.path.splitext(os.path.basename(lora_path))[0]).lower()


def custom_voice_config(config_text: str, speaker_name: str) -> str:
    """config.json of a Base model -> the same model as a one-speaker custom
    voice model (what the ComfyUI pack's Loader reads)."""
    cfg = json.loads(config_text)
    key = speaker_name.lower()
    cfg["tts_model_type"] = "custom_voice"
    cfg["talker_config"]["spk_id"] = {key: SPEAKER_ROW}
    cfg["talker_config"]["spk_is_dialect"] = {key: False}
    return json.dumps(cfg, indent=2, ensure_ascii=False)


def build_batch(text_ids: List[torch.Tensor], codes: List[torch.Tensor], config):
    """Qwen's TTSDataset.collate_fn (qwen-tts finetuning, Apache-2.0): lay text
    and codec channels out the way generate() does in non-streaming mode.
    text_ids: [1, L] "<|im_start|>assistant\\n{text}" ids; codes: [T, 16]."""
    tc = config.talker_config
    lengths = [t.shape[1] + c.shape[0] for t, c in zip(text_ids, codes)]
    b, t = len(codes), max(lengths) + 8
    input_ids = torch.zeros((b, t, 2), dtype=torch.long)
    codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
    text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_mask = torch.zeros((b, t), dtype=torch.bool)
    attention_mask = torch.zeros((b, t), dtype=torch.long)
    codec_0_labels = torch.full((b, t), -100, dtype=torch.long)
    for i, (tid, code) in enumerate(zip(text_ids, codes)):
        lt, lc = tid.shape[1], code.shape[0]
        # text channel: role, 4 pads, bos, text, eos, pads under the audio
        input_ids[i, :3, 0] = tid[0, :3]
        input_ids[i, 3:7, 0] = config.tts_pad_token_id
        input_ids[i, 7, 0] = config.tts_bos_token_id
        input_ids[i, 8:8 + lt - 3, 0] = tid[0, 3:]
        input_ids[i, 8 + lt - 3, 0] = config.tts_eos_token_id
        input_ids[i, 8 + lt - 2:8 + lt + lc, 0] = config.tts_pad_token_id
        text_embedding_mask[i, :8 + lt + lc] = True
        # codec channel: think tokens, speaker slot (6), pads, bos, frames, eos
        input_ids[i, 3:8, 1] = torch.tensor(
            [tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id, 0, tc.codec_pad_id])
        input_ids[i, 8:8 + lt - 3, 1] = tc.codec_pad_id
        input_ids[i, 8 + lt - 3, 1] = tc.codec_pad_id
        input_ids[i, 8 + lt - 2, 1] = tc.codec_bos_id
        input_ids[i, 8 + lt - 1:8 + lt - 1 + lc, 1] = code[:, 0]
        input_ids[i, 8 + lt - 1 + lc, 1] = tc.codec_eos_token_id
        codec_0_labels[i, 8 + lt - 1:8 + lt - 1 + lc] = code[:, 0]
        codec_0_labels[i, 8 + lt - 1 + lc] = tc.codec_eos_token_id
        codec_ids[i, 8 + lt - 1:8 + lt - 1 + lc, :] = code
        codec_embedding_mask[i, 3:8 + lt + lc] = True
        codec_embedding_mask[i, 6] = False  # speaker embedding goes here
        codec_mask[i, 8 + lt - 1:8 + lt - 1 + lc] = True
        attention_mask[i, :8 + lt + lc] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
        "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
        "codec_0_labels": codec_0_labels,
        "codec_ids": codec_ids,
        "codec_mask": codec_mask,
    }


class Qwen3TTSTrainModel(BaseAudioModel):
    arch = "qwen3_tts"
    is_llm = True
    sample_rate = SAMPLE_RATE
    # batches mix clips of any length; get_llm_loss trims the -1 padding
    audio_mixed_length_batches = True

    def __init__(self, device, model_config: ModelConfig, dtype="bf16", custom_pipeline=None,
                 noise_scheduler=None, **kwargs):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_transformer = True
        # matched from the root, then limited to the talker and code-predictor
        # blocks by the network's transformer_only filter
        self.target_lora_modules = ["Qwen3TTSForConditionalGeneration"]
        kw = self.model_config.model_kwargs
        self.speaker_name: Optional[str] = kw.get("speaker_name") or None
        self.ref_audio: Optional[str] = kw.get("ref_audio") or None
        self.recipe = str(kw.get("recipe", "fixed")).lower()
        if self.recipe not in ("fixed", "official"):
            raise ValueError(f"qwen3_tts: recipe must be 'fixed' or 'official', got {self.recipe!r}")
        self.sub_talker_weight = float(kw.get("sub_talker_weight", 0.3))
        self.export_dir: str = str(kw.get("comfyui_export_dir", "") or "")
        self.export_every_save = bool(kw.get("comfyui_export_every_save", True))
        self.export_keep = int(kw.get("comfyui_export_keep", 0))
        self.export_dtype = DTYPES[str(kw.get("comfyui_export_dtype", "fp32")).lower()]
        self.sample_language = kw.get("sample_language", "Auto") or "Auto"
        self.sample_instruct = kw.get("sample_instruct", None)
        self._export_thread = None
        self._speaker: Optional[torch.Tensor] = None  # [1, D] fp32, cpu
        self.tts = None
        self.speech_tokenizer = None
        self.base_path: Optional[str] = None
        self.additional_loss_logs = {}

    @staticmethod
    def get_train_scheduler():
        return None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def load_model(self):
        from .qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

        if self.model_config.quantize:
            print_acc("qwen3_tts: quantization is not supported; ignoring quantize")
        if self.model_config.layer_offloading:
            raise NotImplementedError("Layer offloading is not implemented for qwen3_tts")

        name_or_path = self.model_config.name_or_path or DEFAULT_MODEL
        self.print_and_status_update(f"Loading Qwen3-TTS from {name_or_path}")
        if os.path.isdir(name_or_path):
            self.base_path = name_or_path
        else:
            from huggingface_hub import snapshot_download

            self.base_path = snapshot_download(name_or_path)
        tts = Qwen3TTSModel.from_pretrained(self.base_path, dtype=self.torch_dtype,
                                            device_map=str(self.device_torch), attn_implementation="sdpa")
        model = tts.model
        if model.tts_model_type != "base" or model.speaker_encoder is None:
            raise ValueError(f"qwen3_tts trains a Base model (it needs the speaker encoder); "
                             f"{name_or_path} is a {model.tts_model_type!r} model")
        # the speech tokenizer came in at the training dtype; the pack's data
        # prep (and decoding at fp32) use it in fp32
        model.load_speech_tokenizer(Qwen3TTSTokenizer.from_pretrained(
            os.path.join(self.base_path, "speech_tokenizer"), dtype=torch.float32,
            device_map=str(self.device_torch)))
        model.requires_grad_(False)
        model.eval()

        def enable_gradient_checkpointing():
            model.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        model.enable_gradient_checkpointing = enable_gradient_checkpointing

        self.tts = tts
        self.speech_tokenizer = model.speech_tokenizer
        self.model = model
        self.tokenizer = tts.processor
        self.text_encoder = None
        self.vae = None
        self.pipeline = self
        self.print_and_status_update("Model Loaded")
        if not self.export_dir:
            print_acc("qwen3_tts: comfyui_export_dir is empty, so saves are LoRA files only and will not "
                      "appear in ComfyUI (its Qwen3-TTS nodes cannot load LoRAs). Set comfyui_export_dir in "
                      "the model kwargs, or export a finished LoRA later with "
                      "scripts/export_qwen3_tts_lora.py")

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["talker.model.layers", "talker.code_predictor.model.layers"]

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def get_bucket_divisibility(self):
        return 1

    def save_model(self, output_path, meta, save_dtype):
        raise NotImplementedError("qwen3_tts: only LoRA training is supported (no full-model save)")

    # ------------------------------------------------------------------
    # audio -> codes (latent cache)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, image_list: torch.Tensor, device=None, dtype=None):
        """[1, C, S] waveform at 24 kHz -> DTO carrying [1, T, 16] int32 codes
        (an int extra: float casts in the latent cache would round codes)."""
        if image_list.shape[0] != 1:
            raise ValueError("qwen3_tts encodes one clip at a time: enable Cache Latents for this dataset")
        wav = image_list[0].float().mean(0).cpu().numpy()
        codes = self.speech_tokenizer.encode(wav, sr=SAMPLE_RATE).audio_codes[0]
        codes = codes.to("cpu", torch.int32).contiguous()  # [T, 16] (the tokenizer returns a transposed view)
        return DTO(codes.float()[None], audio_tokens=codes[None])

    def encode_audio(self, audio_data_list):
        return torch.zeros(len(audio_data_list), 1)

    # ------------------------------------------------------------------
    # speaker embedding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _embed_clips(self, paths: List[str]) -> torch.Tensor:
        import librosa
        from .qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        enc = self.model.speaker_encoder
        p = next(enc.parameters())
        embeds = []
        for path in paths:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            mel = mel_spectrogram(torch.from_numpy(audio).unsqueeze(0), n_fft=1024, num_mels=128,
                                  sampling_rate=SAMPLE_RATE, hop_size=256, win_size=1024, fmin=0,
                                  fmax=12000).transpose(1, 2)
            embeds.append(enc(mel.to(p.device, p.dtype)).float().cpu())
        return torch.cat(embeds).mean(0, keepdim=True)

    def _ref_clips(self, source: str) -> List[str]:
        if os.path.isfile(source):
            return [source]
        clips = sorted(f for f in glob.glob(os.path.join(source, "*")) if f.lower().endswith(_AUDIO_EXT))
        if len(clips) > _MAX_REF_CLIPS:
            step = len(clips) / _MAX_REF_CLIPS
            clips = [clips[int(i * step)] for i in range(_MAX_REF_CLIPS)]
        return clips

    def _speaker_embedding(self, batch=None) -> Optional[torch.Tensor]:
        if self._speaker is not None:
            return self._speaker
        source = self.ref_audio
        if source is None and batch is not None and batch.file_items:
            source = os.path.dirname(batch.file_items[0].path)
        if source is None:
            return None
        clips = self._ref_clips(source)
        if not clips:
            raise ValueError(f"qwen3_tts: no audio found for the speaker embedding in {source}")
        self._speaker = self._embed_clips(clips)
        print_acc(f"qwen3_tts: speaker embedding from {len(clips)} clip(s) in {source}")
        return self._speaker

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _text_ids(self, text: str) -> torch.Tensor:
        ids = self.tts.processor(text=self.tts._build_assistant_text(text), return_tensors="pt",
                                 padding=True)["input_ids"]
        ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
        return ids[:, :-5]  # drop the trailing "<|im_end|>\n<|im_start|>assistant\n"

    def get_llm_loss(self, batch) -> torch.Tensor:
        lat = batch.latents
        if not isinstance(lat, DTO) or lat.get("audio_tokens") is None:
            if batch.tensor is None:
                raise ValueError("qwen3_tts got a batch with neither audio nor cached codes")
            lat = self.encode_images(batch.tensor)
        tokens = lat.get("audio_tokens")
        captions = batch.get_caption_list()
        codes, text_ids = [], []
        for i in range(tokens.shape[0]):
            tok = tokens[i].reshape(-1, tokens.shape[-1])  # [T, 16]
            # clips of a batch are padded to the longest with -1 (DTO.stack)
            tok = tok[: int((tok != -1).all(-1).sum())]
            codes.append(tok.long().cpu())
            text_ids.append(self._text_ids(captions[i].strip()))
        b = {k: v.to(self.device_torch) for k, v in build_batch(text_ids, codes, self.model.config).items()}

        model = self.model
        talker = model.talker
        dtype = next(talker.parameters()).dtype
        spk = self._speaker_embedding(batch).to(self.device_torch, dtype)
        ids = b["input_ids"]
        text_embeds = talker.model.text_embedding(ids[:, :, 0])
        # generation always runs text through text_projection; sft_12hz.py
        # only does for the 0.6B model, whose widths differ
        if self.recipe == "fixed" or text_embeds.shape[-1] != talker.config.hidden_size:
            text_embeds = talker.text_projection(text_embeds)
        codec_embeds = talker.model.codec_embedding(ids[:, :, 1]) * b["codec_embedding_mask"]
        codec_embeds[:, 6, :] = spk
        embeds = text_embeds * b["text_embedding_mask"] + codec_embeds
        for i in range(1, 16):
            embeds = embeds + talker.code_predictor.get_input_embeddings()[i - 1](b["codec_ids"][:, :, i]) \
                * b["codec_mask"].unsqueeze(-1)
        labels = b["codec_0_labels"][:, 1:]
        out = talker(inputs_embeds=embeds[:, :-1], attention_mask=b["attention_mask"][:, :-1],
                     labels=labels if self.recipe == "official" else None, output_hidden_states=True)
        hidden = out.hidden_states[0][-1]
        frame_codes = b["codec_ids"][b["codec_mask"]]
        sub_logits, sub_loss = talker.forward_sub_talker_finetune(frame_codes, hidden[b["codec_mask"][:, 1:]])
        if self.recipe == "official":
            talker_loss = out.loss
        else:
            # logits[t] predict labels[t]: the next frame (the causal-LM loss
            # would shift a second time)
            talker_loss = F.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]),
                                          labels.reshape(-1), ignore_index=-100)
            sub_loss = F.cross_entropy(sub_logits.float().reshape(-1, sub_logits.shape[-1]),
                                       frame_codes[:, 1:].reshape(-1))
        loss = talker_loss + self.sub_talker_weight * sub_loss
        self.additional_loss_logs = {"loss/talker": float(talker_loss.detach()),
                                     "loss/sub_talker": float(sub_loss.detach())}
        return loss

    # LoRA keys relative to the Qwen3-TTS root (talker.model.layers.N....);
    # the speaker embedding rides along so the file alone can be exported
    def convert_lora_weights_before_save(self, state_dict):
        out = {}
        for k, v in state_dict.items():
            for prefix in ("transformer.", "diffusion_model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            out[k] = v
        if self._speaker is not None:
            out[SPEAKER_KEY] = self._speaker.clone()
        return out

    def convert_lora_weights_before_load(self, state_dict):
        spk = state_dict.pop(SPEAKER_KEY, None)
        if spk is not None:
            # resume with the speaker embedding the LoRA was trained with
            self._speaker = spk.float().cpu()
        return {(k if k.startswith("transformer.") else "transformer." + k): v for k, v in state_dict.items()}

    # ------------------------------------------------------------------
    # samples: the prompt is the text to speak, in the trained voice
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return self

    @torch.no_grad()
    def generate_single_image(self, pipeline, gen_config: GenerateImageConfig, conditional_embeds,
                              unconditional_embeds, generator, extra):
        import numpy as np
        import torchaudio

        if gen_config.output_ext not in ("mp3", "wav"):
            gen_config.output_ext = "wav"
        text = (gen_config.prompt or "").strip() or "Hello, this is a test."
        # ~15 characters a second of speech; room for slow delivery
        max_tokens = int(min(60.0, max(4.0, len(text) / 15.0) * 2.0) * FRAME_RATE) + 24
        spk = self._speaker_embedding()
        model = self.model
        tc = model.config.talker_config
        weight = model.talker.get_input_embeddings().weight
        saved_row = weight[SPEAKER_ROW].clone()
        saved_type = model.tts_model_type
        spk_id = tc.spk_id if tc.spk_id is not None else {}
        dialect = tc.spk_is_dialect if tc.spk_is_dialect is not None else {}
        torch.manual_seed(int(gen_config.seed) if gen_config.seed is not None else 0)
        was_training = model.training
        model.eval()
        try:
            speaker = ""  # no voice yet (samples before the first step without ref_audio)
            if spk is not None:
                # register the voice the way an exported model carries it
                weight[SPEAKER_ROW] = spk[0].to(weight.device, weight.dtype)
                spk_id[SAMPLE_SPEAKER] = SPEAKER_ROW
                dialect[SAMPLE_SPEAKER] = False
                tc.spk_id, tc.spk_is_dialect = spk_id, dialect
                model.supported_speakers = spk_id.keys()
                speaker = SAMPLE_SPEAKER
            model.tts_model_type = "custom_voice"
            wavs, sr = self.tts.generate_custom_voice(
                text=text, speaker=speaker, language=self.sample_language,
                instruct=self.sample_instruct, non_streaming_mode=True, max_new_tokens=max_tokens)
        finally:
            weight[SPEAKER_ROW] = saved_row
            spk_id.pop(SAMPLE_SPEAKER, None)
            dialect.pop(SAMPLE_SPEAKER, None)
            model.tts_model_type = saved_type
            if was_training:
                model.train()
        wav = torch.from_numpy(np.asarray(wavs[0], dtype=np.float32))
        if wav.ndim == 1:
            wav = wav[None]
        wav = torchaudio.functional.resample(wav, int(sr), SAVE_SAMPLE_RATE)
        return wav[None]  # [1, C, S]; save_image writes image[0] at 48 kHz

    # ------------------------------------------------------------------
    # ComfyUI export: merged custom-voice model folder per save
    # ------------------------------------------------------------------
    def after_network_save(self, lora_path: str, network, is_final: bool):
        if not self.export_dir:
            return
        if not is_final and not self.export_every_save:
            return
        try:
            self._export_merged(lora_path, network, wait=is_final)
        except Exception as e:  # an export failure must never stop training
            print_acc(f"qwen3_tts: ComfyUI export failed: {type(e).__name__}: {e}")

    def _export_merged(self, lora_path: str, network, wait: bool = False):
        import threading

        if self._speaker is None:
            print_acc("qwen3_tts: no speaker embedding yet, skipping the ComfyUI export")
            return
        state = merged_state_dict(self.model, network, self.export_dtype,
                                  os.path.join(self.base_path, "model.safetensors"))
        key = "talker.model.codec_embedding.weight"
        state[key][SPEAKER_ROW] = self._speaker[0].to(state[key].dtype)
        name = self.speaker_name or _speaker_name_from(lora_path)
        self.wait_for_exports()
        thread = threading.Thread(target=self._write_export, args=(lora_path, state, name), daemon=False)
        thread.start()
        self._export_thread = thread
        if wait:
            self.wait_for_exports()

    def wait_for_exports(self):
        if self._export_thread is not None:
            self._export_thread.join()
            self._export_thread = None

    def _write_export(self, lora_path: str, state: dict, speaker_name: str):
        try:
            dest = write_export(state, lora_path, self.base_path, self.export_dir, self.export_keep, print_acc,
                                link_dirs=("speech_tokenizer",),
                                edit_files={"config.json": lambda t: custom_voice_config(t, speaker_name)})
            print_acc(f"qwen3_tts: load it with the Qwen3-TTS Loader (local_model_path = that folder, "
                      f"precision fp32) and the Custom Voice node with custom_speaker_name "
                      f"'{speaker_name}' ({os.path.basename(dest)})")
        except Exception as e:
            print_acc(f"qwen3_tts: ComfyUI export failed: {type(e).__name__}: {e}")
