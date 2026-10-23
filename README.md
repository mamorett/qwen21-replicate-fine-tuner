# qwen21-replicate-fine-tuner

This is a [Cog](https://cog.run) model for fine-tuning and running inference on **Qwen-Image 2.1** (`arch: "qwen_image_2"`) using LoRA adapters.

> **Migration Note:** This repository was converted from a legacy fork of replicate/flux-fine-tuner to exclusively support Qwen-Image 2.1. Legacy FLUX components, LLaVA autocaptioning submodules, and FLUX pipeline patches have been fully purged.

## Features

- **Qwen-Image 2.1 Architecture:** Unified text-to-image and reference-image instruction editing backed by ostris/ai-toolkit pinned at commit `14f8f95090689b585093e5aaf48bc1e2e0a67ea4`.
- **High-VRAM Native BF16 Execution (`low_vram: false`):** The transformer, VAE, and Qwen3-VL text encoder run natively in bf16 on GPU (`cuda:0`), leveraging `cache_text_embeddings: true` and `cache_latents_to_disk: true` for high-throughput training.
- **Native RGBA Transparency Support:** Preserves 4-channel alpha transparency conditioning when `rgba: true` is enabled.
- **Native Autocaptioning:** Built-in VLM captioning powered by Qwen3-VL with automatic VRAM reclamation before training starts.
- **Inference with LoRA:** Fast image generation (1024x1024 to 2048x2048, aspect ratios from 1:1 to 21:9), custom CFG guidance scale, and remote LoRA loading.
- **Automated Validation:** Dual NSFW safety checkers (SD Safety Checker + Falcon NSFW Image Detection) and Weights & Biases telemetry.
- **Hugging Face Hub Integration:** Direct upload of fine-tuned LoRA weights and generated README cards to Hugging Face.

## Repository Setup

Clone recursively to fetch submodules:
```bash
git clone --recursive https://github.com/mamorett/qwen21-replicate-fine-tuner.git
cd qwen21-replicate-fine-tuner
```

Verify submodule status:
```bash
git submodule status
# Expected: 14f8f95090689b585093e5aaf48bc1e2e0a67ea4 ai-toolkit
```

## Running Tests

Install test dependencies and run CPU unit tests:
```bash
pip install -r requirements-test.txt
pytest unit-tests/
```

## Training with Cog

```bash
cog train -i input_images=@images.zip -i steps=2000 -i trigger_word="MY_CHAR"
```

## Inference with Cog

```bash
cog predict -i prompt="MY_CHAR standing in a futuristic metropolis" -i replicate_weights="path/to/lora.safetensors"
```

## Credits

- Training engine powered by [ai-toolkit](https://github.com/ostris/ai-toolkit) by [@ostris](https://github.com/ostris).
- Base model weights: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) and [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).

## License

- The code in this repository is licensed under the [Apache-2.0 License](LICENSE).
- The [ai-toolkit](https://github.com/ostris/ai-toolkit) submodule is licensed under the [MIT License](https://github.com/ostris/ai-toolkit/blob/main/LICENSE).
- Qwen-Image 2.1 is licensed under the [Apache-2.0 License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE).
