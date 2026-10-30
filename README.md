# AI Toolkit: Pipeline Parallelism Fork

A fork of [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit), an all-in-one training suite for image, video and audio diffusion models that runs on consumer hardware, from a web UI or the command line. This fork tracks upstream and merges its changes regularly; everything upstream supports works here too.

**📖 Documentation: [the wiki](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki)**: installation, configuration, datasets, training options, and every fork feature.

## What this fork adds

| Feature | What it does |
|---|---|
| [**Multi-GPU pipeline sharding**](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Multi-GPU-Pipeline-Sharding) | Splits a model's blocks across several GPUs in one process, so models too big for one card can train. UI toggle for SDXL; also works on Krea 2 and MiniMax-H3. |
| [**Text encoder / VAE placement**](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Device-Placement) | `te_device` and `vae_device` move the text encoder or VAE to another GPU. |
| [**Identity Eval**](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Identity-Eval) | Scores face and body likeness against reference photos during training, guards prompt flexibility, and ranks your checkpoints. Shown in a new **Identity** tab. |
| [**Second GPU for sampling**](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Identity-Eval#second-gpu-for-sampling) | `sample.helper_gpu` renders sample rounds on a second GPU. |
| [**Adaptive learning rate**](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Adaptive-LR-and-Early-Stopping) | At each validation, replays the last segment at other learning rates and keeps the best. |
| [**Early stopping**](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Adaptive-LR-and-Early-Stopping#early-stopping) | Stops training when validation loss stops improving. |
| [**Speech models**](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Speech-Models) | LoRA training for OmniVoice and Qwen3-TTS custom voices, with held-out audio validation and merged ComfyUI export. |
| [**MiniMax-H3 teacher matching**](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/MiniMax-H3-Teacher-Matching) | Trains text-only MiniMax-H3 LoRAs against a teacher that sees reference pictures or the clip itself. |
| [**Sampling extras**](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Sampling-Extras) | Sample-only LoRAs with per-LoRA strength, Karras and exponential sigma schedules, LCM in the UI. |
| [**Model browser**](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Model-Browser) | A Browse button on Name or Path fields for picking local models. |

Plus fixes: SDXL sampling with unloaded text encoders, Automagic respecting `max_lr` on resume, `gradient_accumulation` below 1 treated as 1, and nested timers.

## Quick start

```bash
git clone https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism.git ai-toolkit
cd ai-toolkit
```

Then run the **AI Toolkit Manager** for your platform. It installs PyTorch, Python dependencies, Node.js and FFmpeg inside the folder, keeps the install updated, and opens the UI at `http://localhost:8675`:

- **Windows:** `run_windows.bat`
- **Linux** (x86_64 / ARM64): `./run_linux.sh`
- **macOS** (Apple Silicon, experimental): `./run_mac.zsh`

For a manual install, the UI, and how to secure it with `AI_TOOLKIT_AUTH`, see [Installation](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Installation) and [Running the UI](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Running-the-UI).

**Command line:** copy a config from [`config/examples/`](config/examples), edit it, and run:

```bash
python run.py config/my_config.yaml
```

Rerun the same config to resume from the last checkpoint. Don't press Ctrl+C while it is saving. See [Configuration](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Configuration).

## Supported Models

### Image
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (FLUX.1)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (FLUX.2)
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (FLUX.2-klein-base-4B)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (FLUX.2-klein-base-9B)
- [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) (FLUX.3 Image Klein)
- [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) (Flex.1)
- [ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) (Flex.2)
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) (Chroma)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) (Lumina2)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Qwen-Image)
- [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Qwen-Image-2512)
- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Qwen-Image-2.1)
- [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (Qwen-Image-2.1 Turbo)
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) (HiDream I1)
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (OmniGen2)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Z-Image Turbo)
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Z-Image)
- [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) (Z-Image De-Turbo)
- [zhen-nan/L2P](https://huggingface.co/zhen-nan/L2P) (Z-Image L2P)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (SD 1.5)
- [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) (ERNIE-Image)
- [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (Nucleus-Image)
- [Boogu/Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base) (Boogu Image 0.1)
- [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) (HiDream O1)
- [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8) (Ideogram 4 FP8)
- [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) (PRXPixel)
- [speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b) (Iris-3B)
- [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers) (Anima)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo)
- [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design) (Ming-Image 0.1 Design)
- [microsoft/Mage-Flow-Base](https://huggingface.co/microsoft/Mage-Flow-Base) (Mage-Flow)
- [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) (MOSS-Image SFT)
- [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) (MOSS-Image Base)
- [briaai/Fibo-2.0-dev](https://huggingface.co/briaai/Fibo-2.0-dev) (Fibo 2.0 Dev)

### Instruction / Edit
- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (FLUX.1-Kontext-dev)
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) (Qwen-Image-Edit)
- [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Qwen-Image-Edit-2509)
- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Qwen-Image-Edit-2511)
- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Qwen-Image-2.1) - one model for both; it edits when your dataset has control images
- [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (Qwen-Image-2.1 Turbo) - same, with the 8-step turbo weights
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) (HiDream E1)
- [Boogu/Boogu-Image-0.1-Edit](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit) (Boogu Image Edit)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2 Edit Training)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo Edit Training)
- [microsoft/Mage-Flow-Edit-Base](https://huggingface.co/microsoft/Mage-Flow-Edit-Base) (Mage-Flow Edit)

### Video
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (Wan 2.1 1.3B)
- [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) (Wan 2.1 I2V 14B-480P)
- [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) (Wan 2.1 I2V 14B-720P)
- [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (Wan 2.1 14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (Wan 2.2 14B)
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Wan 2.2 I2V 14B)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Wan 2.2 TI2V 5B)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (LTX-2)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2.3)
- [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (LTX-2.5)
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMaxAI/MiniMax-H3)
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMax-H3 Ref2V) - reference image to video

### Audio
- [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (Ace Step 1.5)
- [ACE-Step/acestep-v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base) (Ace Step 1.5 XL)
- [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) (YuE2) - the official audio-to-token encoder is unreleased; training uses the community tokenizer by Kytra ([@sin_ceriously](https://x.com/sin_ceriously)), [Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4).

### LLM
- [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) (Qwen2.5-Omni)

### Experimental
- [lodestones/Zeta-Chroma](https://huggingface.co/lodestones/Zeta-Chroma) (Zeta Chroma)


See [Supported Models](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Supported-Models) for each model's `arch` value and notes. This fork also adds **OmniVoice** and **Qwen3-TTS** ([Speech Models](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/wiki/Speech-Models)).

## Credits and support

AI Toolkit is created by [Ostris](https://github.com/ostris). For help with upstream features, join the [AI Toolkit Discord](https://discord.gg/VXmU2f5WEU). If you find the toolkit useful, consider [supporting Ostris](https://ostris.com/support) or training on [Ostris Cloud](https://cloud.ostris.com).

For problems with fork features, [open an issue here](https://github.com/speedyrulz/ai-toolkit-pipeline-parallelism/issues).
