# Codebase review, 31 Oct 2026

A full pass over the fork (trainer core, data pipeline, model wrappers, samplers,
optimizers, identity eval, audio models, UI) for bugs, VRAM waste and speed.
Everything below under **Fixed** is in the working tree of this branch; the
**Deferred** list is what was found but left alone, with the reason.

Verification done on this machine: `py_compile` of every changed file, ruff on the
bug-class rules (no new findings), the two unit suites (42 tests, all pass), a CPU/GPU
smoke test of the rewritten scheduler, optimizer, float8 and LoRA-hook code, and
`tsc --noEmit` of the UI (clean). No training run was performed; the changes to the
model wrappers are mechanical device/dtype corrections and should be exercised with a
short SDXL + pipeline-sharding job and one Flux-family job before merging.

## Fixed

### Training loop (jobs/process/BaseSDTrainProcess.py, extensions_built_in/sd_trainer/SDTrainer.py)

- `gradient_accumulation_steps > 1` (and `-1`) zeroed the gradients at the start of
  every call, so only the last micro-batch ever reached the optimizer, scaled by 1/N.
  Grads are now zeroed only when a window starts.
- v-prediction loss target used the pixel tensor instead of the latents (shape error, or
  `None` with cached latents).
- `max_grad_norm` was a no-op for optimizers that park bf16 grads in `_accum_grad`
  (Adam8bit, Prodigy8bit, Adafactor, Automagic v1/EXPERIMENT): the grads are restored
  before clipping now, and the OOM path clears the parked grads.
- Only the last micro-batch of a step was cleaned up; all of them are now. With adaptive
  LR the buffered batches keep their latents on the CPU instead of the GPU.
- Reg/non-reg alternation inside an accumulation window never alternated (`batch_step`
  reset inside the loop).
- OOM recovery crashed when `logging.log_every` is null.
- `dadaptation_adam` constructed DAdaptLion.
- A fused optimizer (automagic2, automagic3/4 `fused=True`) with gradient accumulation
  now prints a warning: accumulation and grad clipping do not apply to it.
- `custom_lcm` could not train (`set_timesteps(1000)` always raised); it now honors its
  config and the trainer passes `original_inference_steps`.
- LR scheduler `step` and every diffusers fallback crashed on the injected `total_iters`.
- `mean_flow` loss: the `time_gap.clamp` result was discarded.
- Sample file names: zero-padded index and one shared time stamp, so the samples grid
  keeps prompt order with the second-GPU helper.
- Sample helper failures (OOM on the second card, stale snapshot) no longer end the run:
  its images are rendered on the trainer's GPU and the helper is dropped.
- Whole-model dtype casts (trainer setup, after sampling) are device-only for models whose
  loader keeps fp32 islands (`aitk_cast_on_load = False`, MiniMax-H3).

### Host/device syncs removed from hot paths (speed)

- SNR weighting indexed the SNR table one timestep at a time (one sync per batch element).
- `get_sigmas` in the trainer and in the flow-match / mean-flow schedulers used
  `.nonzero().item()` per element; now one gather.
- Automagic2 learning-rate reporting did one sync per parameter per step.
- `Auto8bitTensor` keeps its scale as a 0-d device tensor (one sync per 8-bit state
  tensor per step removed; old checkpoints with a float scale still load).
- Prodigy8bit accumulated `d_numerator`/`d_denom` with two `.item()` per parameter and
  requantized the constant `p0` every step.
- Stochastic grad accumulation cloned the first gradient instead of taking ownership.
- DiffusionTrainer kept every finished async DB task alive for the whole run.

### Pipeline sharding / multi-GPU placement

- LoRA, DoRA and FullModule forwards multiplied by a network multiplier that lives on the
  first module's GPU: a cross-device error on the first step with `network.type: lora`
  under sharding. The multiplier follows the layer now; LoKr already did this.
- Legacy `StableDiffusion` loader (SD1/SD2/SDXL) ignored `te_device` / `vae_device`.
- `encode_images` / `decode_latents` in `BaseModel` (and Ernie's overrides) computed on
  the caller's device instead of the VAE's own device.
- SDXL: `lora_path` was fused after the UNet was quantized, so the fused LoRA was silently
  lost. It is fused before quantization now.
- Text-encoder LoRA modules follow a text encoder pinned to `te_device`.
- Legacy `save_device_state` crashed on an unloaded text encoder (`FakeTextEncoder`) for
  single-TE models.
- Wrappers that force the VAE onto the training GPU while sending control pixels to
  `vae_device` (flux_kontext, qwen_image_edit, hidream_e1, omnigen2, flux2) and loaders
  that pinned the VAE to the training GPU (chroma, f_light, nucleus, hidream, omnigen2,
  ernie) now use `vae_device`.
- Loaders that moved the text encoder back onto the GPU right after `aitk_post_load`
  parked it (ltx2, z_image, z_image_l2p) respect `low_vram` and `te_device`.
- Sampling pipelines call a new `BaseModel.move_pipeline_for_sampling` instead of
  `pipeline.to(device)`: with a separate `te_device` only the denoiser and VAE move, so
  the text encoder is not dragged onto a full primary card every sample round.
- Float8 (Ostris backend) stored `f8_qdata` as a real float8 buffer, which
  `Module.to(dtype=bf16)` inflated to bf16 (2x weight VRAM). Stored as a uint8 view now.

### Data pipeline (toolkit/data_loader.py, dataloader_mixins.py, data_transfer_object/)

- MiniMax-H3 teacher matching handed a bound model method to every file item; each
  per-fetch deep copy and every spawned worker pickled the whole model. It is a partial of
  a module-level function now.
- JSON datasets lost their captions when `cache_text_embeddings` was on (the caption was
  loaded without the dataset's caption dict).
- `pin_memory: true` was a no-op (the batch object had no `pin_memory` method).
- `num_repeats` encoded and stored N copies of every latent when caching to memory.
- `control_from_same_folder` / `clip_image_from_same_folder` globbed the folder per item
  per fetch; the listing is cached per process.
- `random_crop` with cached latents re-encoded every image on every run for a single
  fixed crop; it falls back to the center crop with a warning.
- `shuffle_tokens` / `random_triggers` with cached embeddings produced one frozen draw and
  a new cache file per run; they are off while caching, like the dropout options.
- Auto-generated controls that are part of the text embedding are now generated before
  the embeddings are cached.
- `alpha_mask` crashed on RGB or palette images; a failed control image silently reused
  the previous control's pixels; wrong attribute when picking the base unconditional
  tensor; caption files are read as `utf-8-sig` so a BOM is not tokenized.

### Samplers

- Dynamic-shift `mu` for 5-D video latents used frames x height instead of
  frames x height x width (LTX2 with `timestep_type: shift`).
- Flow-match `set_train_timesteps` left `sigmas` stale for linear/sigmoid/weighted/
  lognorm_blend (IndexError in `get_sigmas` after the first sample round).
- `MeanFlowScheduler.get_weights_for_timesteps` had the wrong signature and returned a
  float.
- `PromptEmbedsCache.prompts` was a class-level dict shared by every instance.

### Identity eval (toolkit/identity_eval/)

- A worker that fails to launch (bad `evaluator_python`, closed pipe) or a failed
  reference preparation on resume disables identity eval instead of aborting training.
- `to_gpu` replies are tracked, so a failed hand-off is reported instead of surfacing as
  per-image scoring failures; helper offload/onload failures are logged, not fatal.

### Other crashes (undefined names and dead code paths)

- `custom_adapter.condition_noisy_latents`: `control_tensor` referenced before assignment
  (control LoRA training without inpainting input crashed on the first step; also in
  upstream).
- `reference_adapter`: `AttnProcessor2_0` not imported, `network` dereferenced without a
  None check, `args` passed as one positional.
- `photomaker_pipeline`: `rescale_noise_cfg` not imported (`guidance_rescale > 0`).
- `sapiens2`: `np.log` without numpy.
- `ExtractLoraProcess`: `self.dim` never set; inverted `linear_only`.
- `GenerateProcess`: referenced a non-existent `train_config`.
- `TrainESRGANProcess` without a critic; `TrainSliderProcess` prompt-only path
  concatenated a 0-dim timestep.
- DFE v9 broadcast (B,N,C) x (B,1,1,1) into (B,B,N,C) (wrong weighting, B times the
  memory); DFE v3/v7 reset only the last logged loss key.
- `control_generator` used one attribute for BiRefNet and Sapiens2 matting.
- `inference_lora` hook mode silently dropped norm/conv/embedding deltas (they are
  weight-patched and restored on detach now) and dequantized every quantized weight just
  to read its shape.
- `lycoris_utils.rebuild_weight` tested tensors for truthiness.
- `wan22_14b_i2v` with cached latents raises a clear error instead of `NoneType.shape`;
  `flex2` falls back to the item's crop size when `batch.tensor` is None.
- Fuyu captioner loaded the 8B model twice and built the processor with the wrong class.
- Save paths cloned every weight on the GPU before copying it to the CPU.

### UI (ui/)

- Job names are validated as a single path segment (they are used as a folder that is
  deleted recursively).
- `img/caption` and `img/delete` resolve the path before the root check and whitelist the
  caption extension (a `..` path could overwrite or delete files outside the datasets
  root).
- Identity scores on the loss graph are no longer EMA-smoothed and a 0 is kept on the log
  axis.
- Unchecking pipeline sharding clears `pipeline_devices` / `pipeline_balance`.
- A sharded job saves `gpu_ids` as every shard device, and the overview labels the shard
  GPUs and the default scoring GPU.
- Identity suite textareas keep their raw text while typing (a trailing Enter no longer
  vanishes).
- Model browser: an older, slower folder listing can no longer overwrite a newer one.

## Deferred (found, not changed)

Design-level or risky without a training run; listed with the file to look at.

- Bucket batch composition is frozen for the whole run (`dataloader_mixins.py`
  `shuffle_buckets`); fixing it means a main-process batch sampler because of persistent
  workers.
- Latent cache key ignores file content and `standardize_images`
  (`get_latent_info_dict`); adding a file signature invalidates every existing cache once.
- Bf16 merge-in/merge-out for sampling is not round-trip exact (`network_mixins.py`
  `merge_in`/`merge_out`); the frozen base drifts one ulp per sample round. LoKr
  `merge_in` on an Ostris-quantized base requantizes lossily.
- LoRA forward upcasts activations to fp32 and saves the fp32 copy for backward
  (`network_mixins.py` ~309); computing in the activation dtype would save VRAM but
  changes numerics.
- `_get_base_qtype` uses the UNet qtype for text-encoder adapters.
- Hidream/OmniGen2 `low_vram` "move everything back" blocks defeat CPU parking; wrappers'
  `get_prompt_embeds` still pull the text encoder onto the training GPU (needs
  `device=te_device` plumbing through each pipeline's `encode_prompt`).
- Reference/control images are VAE-encoded on every `predict_noise` call in several
  wrappers (qwen_image_edit_plus, flux2, mageflow, flux3, moss, fibo2, ming, boogu,
  wan22_14b_i2v); caching per batch in `condition_noisy_latents` would remove 2-3 encoder
  passes per step.
- qwen_image_edit_plus / omnigen2 shape mismatch under `do_cfg` with control images.
- flux2 `scatter_ids` forces three host syncs per item; position-id tables are rebuilt on
  the CPU per step in several wrappers.
- Textual-inversion embedding restore clones the whole embedding matrix and re-scatters it
  every step (`toolkit/embedding.py`).
- Instant-LoRA mid modules loop over the batch per layer (`toolkit/models/ilora.py`).
- DiffusionTrainer polls SQLite four times per step; loss polling in the UI is 1+N
  requests every 2 s.
- Identity eval: reference templates are rebuilt on every resume (cache dir is write-only),
  scoring workers keep a CUDA context on the trainer's GPU for the whole run, no timeouts
  on prepare/score/render waits, ONNX sessions rebuilt per device move, workers start
  sequentially, background scores land on the wrong wandb step.
- Model browser route lists any directory on the server (by design for a local UI; confine
  to configured roots if the UI is exposed).
- Job stop marks `stopped` before the trainer exits, so the queue can start the next job on
  the same GPU during the final save.
- `sample_step_hook` x0 preview uses the flow formula for DPM++ (preview only).
- PhotoMaker adapter only works for batch size 1; `FluxWithCFGPipeline` leaves the guidance
  bypass on if sampling raises; the unused SDXL `predict_noise` / refiner pipelines call
  `_get_add_time_ids` without `text_encoder_projection_dim`.
- `torch.isfinite(loss)` before backward is one sync per step; left because replacing it
  with `nan_to_num` lets NaN gradients through.
- EMA update is a per-tensor Python loop (`torch._foreach_lerp_` would do it in one call).
