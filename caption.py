import gc
from pathlib import Path
from typing import Optional

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

SYSTEM_PROMPT = (
    "You are an expert image captioner for image generation models. "
    "Generate a concise, clear, natural-language description of the image focusing on subject, "
    "composition, lighting, background, colors, and any visible typography (40-100 words). "
    "Do not use meta-phrases such as 'This image shows' or 'In this photo'."
)


def all_images_are_captioned(image_dir: Path) -> bool:
    image_dir = Path(image_dir)
    image_files = [
        f
        for f in image_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
        and not f.name.startswith(".")
    ]
    if not image_files:
        return True

    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists() or txt_path.stat().st_size == 0:
            return False
    return True


class Captioner:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None

    def all_images_are_captioned(self, image_dir: Path) -> bool:
        return all_images_are_captioned(image_dir)

    def load_models(self):
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"Loading autocaptioning model {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    def caption_image(
        self,
        image_path: Path,
        trigger_word: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        rgba: bool = False,
    ) -> str:
        if self.model is None or self.processor is None:
            self.load_models()

        import numpy as np
        import torch
        from PIL import Image

        with Image.open(image_path) as raw_img:
            img = raw_img.convert("RGBA" if rgba else "RGB")
            is_transparent = False
            if rgba and raw_img.mode == "RGBA":
                alpha = np.array(raw_img.getchannel("A"))
                if alpha.min() < 255:
                    is_transparent = True

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                        + "\nProvide only the caption without prefix or introduction.",
                    },
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input],
            images=[img.convert("RGB")],
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            caption = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        # Build final caption
        parts = []
        if is_transparent:
            parts.append("This is an RGBA image with transparency.")

        if prefix:
            prefix_str = prefix.strip()
            if trigger_word and trigger_word not in prefix_str:
                prefix_str = f"{trigger_word} {prefix_str}".strip()
            parts.append(prefix_str)
        elif trigger_word and trigger_word not in caption:
            parts.append(trigger_word)

        parts.append(caption)

        if is_transparent:
            parts.append(
                "The image has alpha channel and the background is transparent."
            )

        if suffix:
            suffix_str = suffix.strip()
            if trigger_word and trigger_word not in suffix_str and not prefix:
                suffix_str = f"{suffix_str} {trigger_word}".strip()
            parts.append(suffix_str)

        final_caption = " ".join([p for p in parts if p]).strip()
        # Clean double spaces
        while "  " in final_caption:
            final_caption = final_caption.replace("  ", " ")
        return final_caption

    def caption_images(
        self,
        image_dir: Path,
        trigger_word: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        rgba: bool = False,
    ):
        image_dir = Path(image_dir)
        image_files = sorted(
            [
                f
                for f in image_dir.iterdir()
                if f.is_file()
                and f.suffix.lower() in IMAGE_EXTENSIONS
                and not f.name.startswith(".")
            ]
        )

        for img_path in image_files:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists() and txt_path.stat().st_size > 0:
                continue

            caption = self.caption_image(
                img_path,
                trigger_word=trigger_word,
                prefix=prefix,
                suffix=suffix,
                rgba=rgba,
            )
            txt_path.write_text(caption, encoding="utf-8")

        # Mandatory cleanup
        self.cleanup()

    def cleanup(self):
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def caption_images(
    image_dir: Path,
    trigger_word: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    rgba: bool = False,
):
    captioner = Captioner()
    captioner.caption_images(
        image_dir,
        trigger_word=trigger_word,
        prefix=prefix,
        suffix=suffix,
        rgba=rgba,
    )
