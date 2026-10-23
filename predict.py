import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure ai-toolkit is in sys.path
sys.path.insert(0, str(Path("ai-toolkit").resolve()))

import numpy as np
import torch

try:
    from cog import BasePredictor, Input, Path as CogPath
except ImportError:

    class BasePredictor:
        pass

    def Input(**kwargs):  # noqa: N802
        return kwargs.get("default")

    CogPath = Path
from PIL import Image
from transformers import (
    AutoModelForImageClassification,
    CLIPImageProcessor,
    ViTImageProcessor,
)

from weights import (
    WeightsDownloadCache,
    download_archive,
    ensure_base_weights,
)

SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
SAFETY_CACHE_PATH = Path("safety-cache")
FEATURE_EXTRACTOR = Path("feature-extractor")

FALCON_MODEL_NAME = "Falconsai/nsfw_image_detection"
FALCON_MODEL_CACHE = "falcon-cache"
FALCON_MODEL_URL = (
    "https://weights.replicate.delivery/default/falconai/nsfw-image-detection.tar"
)

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (928, 1152),
    "5:4": (1152, 928),
    "21:9": (1536, 640),
    "9:21": (640, 1536),
}

logging.getLogger("diffusers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)


def round_to_16(val: int) -> int:
    return max(512, min(2048, int(round(val / 16.0) * 16)))


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load Qwen-Image 2.1 model and safety checkers into memory."""
        start = time.time()
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["DISABLE_TELEMETRY"] = "1"

        self.weights_cache = WeightsDownloadCache()
        self.loaded_lora_url: Optional[str] = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize safety checkers
        print("Loading safety checkers...")
        try:
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )

            if not SAFETY_CACHE_PATH.exists():
                download_archive(SAFETY_URL, SAFETY_CACHE_PATH)
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                SAFETY_CACHE_PATH, torch_dtype=torch.float16
            ).to(self.device)
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                FEATURE_EXTRACTOR
            )
        except Exception as e:
            print(f"Warning: could not load SD safety checker: {e}")
            self.safety_checker = None
            self.feature_extractor = None

        try:
            if not Path(FALCON_MODEL_CACHE).exists():
                download_archive(FALCON_MODEL_URL, Path(FALCON_MODEL_CACHE))
            self.falcon_model = AutoModelForImageClassification.from_pretrained(
                FALCON_MODEL_NAME,
                cache_dir=FALCON_MODEL_CACHE,
            ).to(self.device)
            self.falcon_processor = ViTImageProcessor.from_pretrained(FALCON_MODEL_NAME)
        except Exception as e:
            print(f"Warning: could not load Falcon safety checker: {e}")
            self.falcon_model = None
            self.falcon_processor = None

        print("Ensuring Qwen-Image 2.1 base weights...")
        ensure_base_weights()

        print("Loading Qwen-Image 2.1 pipeline...")
        self.pipe = None
        # Prefer Diffusers QwenImage21Pipeline if present
        try:
            from diffusers import QwenImage21Pipeline

            self.pipe = QwenImage21Pipeline.from_pretrained(
                "Qwen/Qwen-Image-2.1",
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.is_diffusers_pipe = True
            print("Successfully loaded diffusers.QwenImage21Pipeline")
        except Exception as e:
            print(
                f"diffusers.QwenImage21Pipeline not available: {e}, falling back to ai-toolkit pipeline"
            )
            try:
                from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import (
                    QwenImage2Model,
                )
                from toolkit.config_modules import ModelConfig

                model_config = ModelConfig(
                    name_or_path="Comfy-Org/Qwen-Image-2.1",
                    arch="qwen_image_2",
                    quantize=False,
                    low_vram=False,
                    model_kwargs={"rgba": False},
                )
                model_inst = QwenImage2Model(self.device, model_config, dtype="bf16")
                model_inst.load_model()
                self.pipe = model_inst.pipeline
                self.model_inst = model_inst
                self.is_diffusers_pipe = False
                print("Successfully loaded ai-toolkit QwenImage2Model pipeline")
            except Exception as ex:
                print(f"Error loading pipeline: {ex}")

        print(f"Setup completed in {time.time() - start:.2f}s")

    def load_lora(self, lora_url: str, lora_scale: float = 1.0):
        if not self.pipe:
            return
        if self.loaded_lora_url == lora_url:
            if self.is_diffusers_pipe and hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters(["main"], adapter_weights=[lora_scale])
            return

        # Unload existing LoRA
        if self.is_diffusers_pipe and hasattr(self.pipe, "unload_lora_weights"):
            self.pipe.unload_lora_weights()

        print(f"Downloading and loading LoRA weights from: {lora_url}")
        lora_path = self.weights_cache.ensure(lora_url)

        if self.is_diffusers_pipe and hasattr(self.pipe, "load_lora_weights"):
            self.pipe.load_lora_weights(str(lora_path), adapter_name="main")
            if hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters(["main"], adapter_weights=[lora_scale])
        self.loaded_lora_url = lora_url

    def run_safety_checker(
        self, images: List[Image.Image]
    ) -> Tuple[List[Image.Image], List[bool]]:
        if self.safety_checker is None or self.feature_extractor is None:
            return images, [False] * len(images)

        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(
            self.device
        )
        np_images = [np.array(img.convert("RGB")) for img in images]
        checked_images, has_nsfw = self.safety_checker(
            images=np_images,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return images, has_nsfw

    def run_falcon_safety_checker(self, image: Image.Image) -> bool:
        if self.falcon_model is None or self.falcon_processor is None:
            return True
        with torch.no_grad():
            inputs = self.falcon_processor(
                images=image.convert("RGB"), return_tensors="pt"
            ).to(self.device)
            outputs = self.falcon_model(**inputs)
            predicted_label = outputs.logits.argmax(-1).item()
            result = self.falcon_model.config.id2label[predicted_label]
        return result == "normal"

    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation."),
        negative_prompt: str = Input(
            default="",
            description="Negative prompt. Only active when guidance_scale > 1.0.",
        ),
        image: CogPath = Input(
            default=None,
            description="Optional reference image for Qwen-Image 2.1 instruction-based editing.",
        ),
        aspect_ratio: str = Input(
            default="1:1",
            choices=[
                "1:1",
                "16:9",
                "9:16",
                "4:3",
                "3:4",
                "3:2",
                "2:3",
                "4:5",
                "5:4",
                "21:9",
                "9:21",
                "custom",
            ],
            description="Aspect ratio for generated image.",
        ),
        width: int = Input(
            default=1024,
            ge=512,
            le=2048,
            description="Width of generated image (used when aspect_ratio is custom). Must be multiple of 16.",
        ),
        height: int = Input(
            default=1024,
            ge=512,
            le=2048,
            description="Height of generated image (used when aspect_ratio is custom). Must be multiple of 16.",
        ),
        num_outputs: int = Input(
            default=1,
            ge=1,
            le=4,
            description="Number of images to generate.",
        ),
        num_inference_steps: int = Input(
            default=28,
            ge=1,
            le=50,
            description="Number of denoising steps (recommended 20-40).",
        ),
        guidance_scale: float = Input(
            default=1.0,
            ge=1.0,
            le=10.0,
            description="Guidance scale / true_cfg_scale. 1.0 is distilled default (no CFG overhead).",
        ),
        rgba: bool = Input(
            default=False,
            description="Generate 4-channel transparent RGBA output (forces PNG output format).",
        ),
        replicate_weights: str = Input(
            default=None,
            description="URL or Replicate reference to trained LoRA weights.",
        ),
        lora_scale: float = Input(
            default=1.0,
            ge=-1.0,
            le=2.0,
            description="LoRA adapter scale.",
        ),
        seed: int = Input(
            default=None,
            description="Random seed. Leave blank to randomize.",
        ),
        output_format: str = Input(
            default="webp",
            choices=["webp", "jpg", "png"],
            description="Output format for generated images.",
        ),
        output_quality: int = Input(
            default=90,
            ge=1,
            le=100,
            description="Quality when saving output images (1-100).",
        ),
        disable_safety_checker: bool = Input(
            default=False,
            description="Disable NSFW safety checkers.",
        ),
    ) -> List[CogPath]:
        """Run a single prediction on the model."""
        if aspect_ratio != "custom" and aspect_ratio in ASPECT_RATIOS:
            w, h = ASPECT_RATIOS[aspect_ratio]
        else:
            w = round_to_16(width)
            h = round_to_16(height)

        if rgba:
            output_format = "png"

        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if replicate_weights:
            self.load_lora(replicate_weights, lora_scale=lora_scale)

        ref_image = None
        if image:
            ref_image = Image.open(str(image)).convert("RGBA" if rgba else "RGB")

        outputs: List[Image.Image] = []

        if self.is_diffusers_pipe and self.pipe:
            call_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt if guidance_scale > 1.0 else None,
                "width": w,
                "height": h,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
            }
            # Handle true_cfg_scale / guidance_scale
            if "true_cfg_scale" in self.pipe.__call__.__code__.co_varnames:
                call_kwargs["true_cfg_scale"] = guidance_scale
            else:
                call_kwargs["guidance_scale"] = guidance_scale

            if ref_image and "image" in self.pipe.__call__.__code__.co_varnames:
                call_kwargs["image"] = ref_image

            for _ in range(num_outputs):
                result = self.pipe(**call_kwargs)
                outputs.append(result.images[0])
        elif hasattr(self, "model_inst") and self.model_inst:
            # Use ai-toolkit sampling pipeline
            for _ in range(num_outputs):
                cond_embeds = self.model_inst.prompt_encoder(
                    prompt,
                    negative_prompt=negative_prompt if guidance_scale > 1.0 else None,
                )
                cond_images = [ref_image] if ref_image else None
                gen_images = self.pipe(
                    conditional_embeds=cond_embeds,
                    width=w,
                    height=h,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    condition_images=cond_images,
                )
                outputs.append(gen_images[0])
        else:
            raise RuntimeError("Pipeline is not initialized.")

        # Safety checking
        output_paths: List[CogPath] = []
        out_dir = Path("/tmp/predictions")
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(outputs):
            is_nsfw = False
            if not disable_safety_checker:
                _, has_nsfw_concepts = self.run_safety_checker([img])
                is_normal_falcon = self.run_falcon_safety_checker(img)
                if has_nsfw_concepts[0] or not is_normal_falcon:
                    is_nsfw = True
                    print(f"Image {i} flagged as NSFW!")

            if is_nsfw:
                # Return placeholder or warning
                black_img = Image.new(
                    "RGBA" if rgba else "RGB", (w, h), (0, 0, 0, 255 if rgba else 0)
                )
                img = black_img

            out_path = out_dir / f"output_{i}.{output_format}"
            if output_format == "png":
                img.save(str(out_path), format="PNG")
            else:
                img.convert("RGB").save(
                    str(out_path), format=output_format.upper(), quality=output_quality
                )
            output_paths.append(CogPath(out_path))

        return output_paths
