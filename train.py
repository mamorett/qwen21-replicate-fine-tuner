import shutil
import sys
import tarfile
import zipfile
from collections import OrderedDict
from string import Template
from typing import Optional, Sequence, Union

from cog import BaseModel, Input, Path, Secret

# Ensure ai-toolkit is in sys.path
sys.path.insert(0, str(Path("ai-toolkit").resolve()))


from caption import Captioner, all_images_are_captioned
from wandb_client import WeightsAndBiasesClient, logout_wandb
from weights import ensure_base_weights

INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output")
JOB_NAME = "qwen_image_21_lora"
JOB_DIR = OUTPUT_DIR / JOB_NAME


class TrainingOutput(BaseModel):
    weights: Path


def parse_resolution(resolution: Union[str, Sequence[int]]) -> list[int]:
    if isinstance(resolution, str):
        parts = [p.strip() for p in resolution.split(",") if p.strip()]
        res_list = [int(p) for p in parts]
    else:
        res_list = [int(p) for p in resolution]
    if not res_list:
        raise ValueError("At least one resolution must be specified.")
    for res in res_list:
        if res < 512 or res > 2048:
            raise ValueError(f"Resolution {res} must be between 512 and 2048.")
        if res % 16 != 0:
            raise ValueError(f"Resolution {res} must be divisible by 16.")
    return res_list


def build_train_config(
    trigger_word: Optional[str] = None,
    steps: int = 2000,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    resolution: Union[str, Sequence[int]] = "1024",
    lora_rank: int = 16,
    caption_dropout_rate: float = 0.05,
    optimizer: str = "adamw8bit",
    weight_decay: float = 1e-4,
    rgba: bool = False,
    cache_latents_to_disk: bool = True,
    gradient_checkpointing: bool = True,
    low_vram: bool = False,
    quantize_te: bool = False,
    gradient_accumulation: int = 1,
    wandb_api_key: Optional[str] = None,
    wandb_sample_prompts: Optional[str] = None,
    wandb_sample_interval: int = 500,
    wandb_save_interval: int = 500,
    training_folder: str = "output",
    folder_path: str = "input_images",
) -> OrderedDict:
    res_list = parse_resolution(resolution)
    has_wandb_sampling = bool(wandb_api_key and wandb_sample_prompts)
    sample_prompts = []
    if has_wandb_sampling and wandb_sample_prompts:
        sample_prompts = [
            p.strip()
            for p in wandb_sample_prompts.replace("\r", "").split("\n")
            if p.strip()
        ]

    save_every = wandb_save_interval if wandb_api_key else 500
    sample_every = wandb_sample_interval if has_wandb_sampling else 2000
    disable_sampling = not has_wandb_sampling

    process_config = OrderedDict(
        [
            ("type", "diffusion_trainer"),
            ("training_folder", training_folder),
            ("device", "cuda:0"),
            ("trigger_word", trigger_word if trigger_word else None),
            ("performance_log_every", 10),
            (
                "network",
                OrderedDict(
                    [
                        ("type", "lora"),
                        ("linear", lora_rank),
                        ("linear_alpha", lora_rank),
                        ("conv", 0),
                        ("conv_alpha", 0),
                        (
                            "network_kwargs",
                            OrderedDict(
                                [
                                    ("ignore_if_contains", []),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            (
                "save",
                OrderedDict(
                    [
                        ("dtype", "bf16"),
                        ("save_every", save_every),
                        ("max_step_saves_to_keep", 2),
                        ("save_format", "diffusers"),
                        ("push_to_hub", False),
                    ]
                ),
            ),
            (
                "datasets",
                [
                    OrderedDict(
                        [
                            ("folder_path", folder_path),
                            ("caption_ext", "txt"),
                            ("caption_dropout_rate", caption_dropout_rate),
                            ("cache_latents_to_disk", cache_latents_to_disk),
                            ("resolution", res_list),
                            ("num_repeats", 1),
                        ]
                    )
                ],
            ),
            (
                "train",
                OrderedDict(
                    [
                        ("batch_size", batch_size),
                        ("steps", steps),
                        ("gradient_accumulation", gradient_accumulation),
                        ("train_unet", True),
                        ("train_text_encoder", False),
                        ("gradient_checkpointing", gradient_checkpointing),
                        ("noise_scheduler", "flowmatch"),
                        ("timestep_type", "shift"),
                        ("optimizer", optimizer),
                        (
                            "optimizer_params",
                            OrderedDict(
                                [
                                    ("weight_decay", weight_decay),
                                ]
                            ),
                        ),
                        ("lr", learning_rate),
                        ("dtype", "bf16"),
                        ("cache_text_embeddings", True),
                        ("unload_text_encoder", False),
                        ("disable_sampling", disable_sampling),
                        ("skip_first_sample", True),
                    ]
                ),
            ),
            (
                "model",
                OrderedDict(
                    [
                        ("name_or_path", "Comfy-Org/Qwen-Image-2.1"),
                        ("arch", "qwen_image_2"),
                        ("quantize", False),
                        ("quantize_te", quantize_te),
                        ("qtype_te", "convrot8"),
                        ("low_vram", low_vram),
                        (
                            "model_kwargs",
                            OrderedDict(
                                [
                                    ("rgba", rgba),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            (
                "sample",
                OrderedDict(
                    [
                        ("sampler", "flowmatch"),
                        ("sample_every", sample_every),
                        ("width", 1024),
                        ("height", 1024),
                        ("guidance_scale", 1.0),
                        ("sample_steps", 20),
                        ("prompts", sample_prompts),
                    ]
                ),
            ),
        ]
    )

    return OrderedDict(
        [
            ("job", "extension"),
            (
                "config",
                OrderedDict(
                    [
                        ("name", "qwen_image_21_lora"),
                        ("process", [process_config]),
                    ]
                ),
            ),
            (
                "meta",
                OrderedDict(
                    [
                        ("name", "qwen_image_21_lora"),
                        ("version", "1.0"),
                    ]
                ),
            ),
        ]
    )


def extract_zip(input_images: Path, input_dir: Path):
    if not zipfile.is_zipfile(input_images):
        raise ValueError("input_images must be a zip file")

    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(input_images, "r") as zf:
        for member in zf.infolist():
            parts = member.filename.split("/")
            if any(p.startswith("__MACOSX") or p.startswith("._") for p in parts):
                continue
            filename = Path(member.filename).name
            if not filename or filename.startswith("._"):
                continue
            target_path = input_dir / filename
            with zf.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    print(f"Extracted {count} files from zip to {input_dir}")


def clean_up():
    logout_wandb()
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    tar_path = Path("trained_model.tar")
    if tar_path.exists():
        tar_path.unlink()


def handle_hf_readme(
    hf_repo_id: str,
    trigger_word: Optional[str],
    steps: int,
    learning_rate: float,
    lora_rank: int,
):
    readme_path = OUTPUT_DIR / "README.md"
    readme_template_path = Path("hugging-face-readme-template.md")

    template_content = readme_template_path.read_text()
    title = (
        hf_repo_id.split("/")[1].replace("-", " ").title()
        if "/" in hf_repo_id
        else hf_repo_id
    )

    trigger_section = (
        f"\n## Trigger words\n\nYou should use `{trigger_word}` to trigger the image generation.\n"
        if trigger_word
        else ""
    )
    instance_prompt = f"instance_prompt: {trigger_word}" if trigger_word else ""
    training_details = f"\n## Training details\n\n- Steps: {steps}\n- Learning rate: {learning_rate}\n- LoRA rank: {lora_rank}\n"

    variables = {
        "repo_id": hf_repo_id,
        "title": title,
        "trigger_word": trigger_word or "",
        "trigger_section": trigger_section,
        "instance_prompt": instance_prompt,
        "training_details": training_details,
    }

    readme_path.write_text(Template(template_content).substitute(variables))


_ACTIVE_WANDB_CLIENT: Optional[WeightsAndBiasesClient] = None


def setup_custom_trainer():
    try:
        from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
        from toolkit.config_modules import GenerateImageConfig
        import toolkit.extension

        class CustomDiffusionTrainer(DiffusionTrainer):
            def cache_sample_prompts(self):
                """Fix Issue #1059: pass width and height into GenerateImageConfig."""
                if self.train_config.disable_sampling:
                    return
                if (
                    self.sample_config is not None
                    and self.sample_config.samples is not None
                    and len(self.sample_config.samples) > 0
                ):
                    self.sd.sample_prompts_cache = []
                    sample_folder = Path(self.save_root) / "samples"
                    output_path = str(sample_folder / "test.jpg")
                    for i in range(len(self.sample_config.prompts)):
                        sample_item = self.sample_config.samples[i]
                        prompt = self.sample_config.prompts[i]
                        if self.trigger_word is not None:
                            prompt = self.sd.inject_trigger_into_prompt(
                                prompt, self.trigger_word, add_if_not_present=False
                            )
                        gen_img_config = GenerateImageConfig(
                            prompt=prompt,
                            negative_prompt=sample_item.neg,
                            output_path=output_path,
                            width=sample_item.width,
                            height=sample_item.height,
                            ctrl_img=sample_item.ctrl_img,
                            ctrl_img_1=sample_item.ctrl_img_1,
                            ctrl_img_2=sample_item.ctrl_img_2,
                            ctrl_img_3=sample_item.ctrl_img_3,
                        )
                        has_control_images = any(
                            [
                                gen_img_config.ctrl_img is not None,
                                gen_img_config.ctrl_img_1 is not None,
                                gen_img_config.ctrl_img_2 is not None,
                                gen_img_config.ctrl_img_3 is not None,
                            ]
                        )
                        if (
                            self.sd.encode_control_in_text_embeddings
                            and has_control_images
                        ):
                            self.sd.prepare_sample_prompt_context(gen_img_config)
                        self.sd.sample_prompts_cache.append(gen_img_config)

            def hook_train_loop(self, batch):
                super().hook_train_loop(batch)
                global _ACTIVE_WANDB_CLIENT
                if _ACTIVE_WANDB_CLIENT is not None and hasattr(self, "loss"):
                    loss_val = float(self.loss) if self.loss is not None else 0.0
                    _ACTIVE_WANDB_CLIENT.log_loss(
                        {"loss": loss_val}, step=self.step_num
                    )

            def sample(self, step=None, is_first=False):
                super().sample(step=step, is_first=is_first)
                global _ACTIVE_WANDB_CLIENT
                if _ACTIVE_WANDB_CLIENT is not None:
                    sample_folder = Path(self.save_root) / "samples"
                    if sample_folder.exists():
                        step_str = str(step) if step is not None else ""
                        sample_images = sorted(
                            sample_folder.glob(f"*{step_str}*.jpg")
                        ) or sorted(sample_folder.glob("*.jpg"))
                        if sample_images:
                            _ACTIVE_WANDB_CLIENT.log_samples(sample_images, step=step)

            def post_save_hook(self, save_path: str):
                super().post_save_hook(save_path)
                global _ACTIVE_WANDB_CLIENT
                if (
                    _ACTIVE_WANDB_CLIENT is not None
                    and save_path
                    and Path(save_path).exists()
                ):
                    _ACTIVE_WANDB_CLIENT.save_weights(Path(save_path))

        orig_process_dict_fn = toolkit.extension.get_all_extensions_process_dict

        def patched_process_dict():
            pdict = orig_process_dict_fn()
            pdict["diffusion_trainer"] = CustomDiffusionTrainer
            return pdict

        toolkit.extension.get_all_extensions_process_dict = patched_process_dict
    except Exception as e:
        print(f"Note: setup_custom_trainer initialization: {e}")


def train(
    input_images: Path = Input(
        description="Zip file containing training images (.png, .jpg, .webp) and optional .txt captions."
    ),
    trigger_word: str = Input(
        default=None,
        description="Optional trigger word (e.g. 'MY_CHAR').",
    ),
    steps: int = Input(
        default=2000,
        ge=3,
        le=6000,
        description="Number of training steps.",
    ),
    learning_rate: float = Input(
        default=1e-4,
        ge=1e-6,
        le=1e-2,
        description="Learning rate.",
    ),
    batch_size: int = Input(
        default=1,
        ge=1,
        le=16,
        description="Batch size.",
    ),
    resolution: str = Input(
        default="1024",
        description="Comma-separated resolutions (e.g. '1024' or '1024,1536,2048').",
    ),
    lora_rank: int = Input(
        default=16,
        ge=4,
        le=128,
        description="Sets both network.linear and network.linear_alpha.",
    ),
    caption_dropout_rate: float = Input(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Caption dropout rate.",
    ),
    optimizer: str = Input(
        default="adamw8bit",
        choices=["adamw8bit", "adamw", "prodigy", "lion"],
        description="Optimizer type.",
    ),
    weight_decay: float = Input(
        default=1e-4,
        ge=0.0,
        le=1.0,
        description="Weight decay for the optimizer.",
    ),
    rgba: bool = Input(
        default=False,
        description="Preserve alpha channel for transparent PNG asset training.",
    ),
    cache_latents_to_disk: bool = Input(
        default=True,
        description="Cache VAE latents to disk.",
    ),
    gradient_checkpointing: bool = Input(
        default=True,
        description="Enable gradient checkpointing.",
    ),
    low_vram: bool = Input(
        default=False,
        description="Keep False by default for maximum training speed on server GPUs.",
    ),
    quantize_te: bool = Input(
        default=False,
        description="Keep False by default (bf16 Qwen3-VL); if True, enables convrot8 on the text encoder.",
    ),
    autocaption: bool = Input(
        default=True,
        description="Automatically caption images missing .txt files.",
    ),
    autocaption_prefix: str = Input(
        default=None,
        description="Optional prefix for autocaptioning.",
    ),
    autocaption_suffix: str = Input(
        default=None,
        description="Optional suffix for autocaptioning.",
    ),
    hf_repo_id: str = Input(
        default=None,
        description="Hugging Face repo ID to upload model.",
    ),
    hf_token: Secret = Input(
        default=None,
        description="Hugging Face API token.",
    ),
    wandb_api_key: Secret = Input(
        default=None,
        description="Weights & Biases API key.",
    ),
    wandb_project: str = Input(
        default="qwen_image_21_lora",
        description="Weights & Biases project name.",
    ),
    wandb_run: str = Input(
        default=None,
        description="Weights & Biases run name.",
    ),
    wandb_entity: str = Input(
        default=None,
        description="Weights & Biases entity.",
    ),
    wandb_sample_interval: int = Input(
        default=500,
        description="Steps interval for W&B sampling.",
    ),
    wandb_sample_prompts: str = Input(
        default=None,
        description="Prompts for W&B sampling (newline separated).",
    ),
    wandb_save_interval: int = Input(
        default=500,
        description="Steps interval for intermediate LoRA saves.",
    ),
    skip_training_and_use_pretrained_hf_lora_url: str = Input(
        default=None,
        description="URL to existing LoRA to package without training.",
    ),
) -> TrainingOutput:
    clean_up()
    output_tar = Path("trained_model.tar")

    # Fast-path: Skip training and use pre-trained LoRA
    if skip_training_and_use_pretrained_hf_lora_url:
        print(
            f"Skipping training, downloading: {skip_training_and_use_pretrained_hf_lora_url}"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        lora_target = OUTPUT_DIR / "lora.safetensors"
        from weights import download_weights

        download_weights(skip_training_and_use_pretrained_hf_lora_url, lora_target)
        with tarfile.open(output_tar, "w") as tar:
            tar.add(lora_target, arcname="lora.safetensors")
        return TrainingOutput(weights=Path(output_tar))

    # Pre-warm base weights
    ensure_base_weights(quantize_te=quantize_te)

    # Extract input images
    extract_zip(Path(input_images), INPUT_DIR)

    # Autocaption if requested
    if autocaption and not all_images_are_captioned(INPUT_DIR):
        print("Autocaptioning images...")
        captioner = Captioner()
        captioner.caption_images(
            INPUT_DIR,
            trigger_word=trigger_word,
            prefix=autocaption_prefix,
            suffix=autocaption_suffix,
            rgba=rgba,
        )

    # W&B initialization
    global _ACTIVE_WANDB_CLIENT
    wandb_key = wandb_api_key.get_secret_value() if wandb_api_key else None
    if wandb_key:
        sample_prompts_list = (
            [p.strip() for p in wandb_sample_prompts.split("\n") if p.strip()]
            if wandb_sample_prompts
            else []
        )
        _ACTIVE_WANDB_CLIENT = WeightsAndBiasesClient(
            api_key=wandb_key,
            project=wandb_project,
            config={
                "steps": steps,
                "learning_rate": learning_rate,
                "lora_rank": lora_rank,
                "batch_size": batch_size,
                "resolution": resolution,
                "rgba": rgba,
                "trigger_word": trigger_word,
            },
            sample_prompts=sample_prompts_list,
            entity=wandb_entity,
            name=wandb_run,
        )

    # Build config & setup trainer
    setup_custom_trainer()
    train_config = build_train_config(
        trigger_word=trigger_word,
        steps=steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        resolution=resolution,
        lora_rank=lora_rank,
        caption_dropout_rate=caption_dropout_rate,
        optimizer=optimizer,
        weight_decay=weight_decay,
        rgba=rgba,
        cache_latents_to_disk=cache_latents_to_disk,
        gradient_checkpointing=gradient_checkpointing,
        low_vram=low_vram,
        quantize_te=quantize_te,
        wandb_api_key=wandb_key,
        wandb_sample_prompts=wandb_sample_prompts,
        wandb_sample_interval=wandb_sample_interval,
        wandb_save_interval=wandb_save_interval,
        training_folder=str(OUTPUT_DIR),
        folder_path=str(INPUT_DIR),
    )

    # Execute training via ai-toolkit
    from toolkit.job import run_job

    run_job(train_config)

    if _ACTIVE_WANDB_CLIENT:
        _ACTIVE_WANDB_CLIENT.finish()
        _ACTIVE_WANDB_CLIENT = None

    # Output Packaging
    final_lora = JOB_DIR / f"{JOB_NAME}.safetensors"
    if not final_lora.exists():
        candidates = list(JOB_DIR.glob(f"{JOB_NAME}*.safetensors"))
        unstepped = [
            c
            for c in candidates
            if not any(char.isdigit() for char in c.stem.split("_")[-1])
        ]
        final_lora = (
            unstepped[0] if unstepped else (candidates[0] if candidates else None)
        )

    if final_lora and final_lora.exists():
        dest_lora = OUTPUT_DIR / "lora.safetensors"
        shutil.move(str(final_lora), str(dest_lora))
    else:
        raise RuntimeError(
            "Training finished but no LoRA safetensors file was generated."
        )

    # Remove stepped checkpoints and optimizer
    for p in list(JOB_DIR.glob("*.safetensors")) + list(
        OUTPUT_DIR.glob(f"{JOB_NAME}_*.safetensors")
    ):
        if p.exists() and p.name != "lora.safetensors":
            p.unlink()

    opt_pt = JOB_DIR / "optimizer.pt"
    if opt_pt.exists():
        opt_pt.unlink()

    # Clean intermediate caches and samples
    samples_dir = JOB_DIR / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)

    # Copy captions
    captions_dest = OUTPUT_DIR / "captions"
    captions_dest.mkdir(parents=True, exist_ok=True)
    for cap_file in INPUT_DIR.glob("*.txt"):
        shutil.copy(cap_file, captions_dest)

    # Hugging Face upload if requested
    hf_token_val = hf_token.get_secret_value() if hf_token else None
    if hf_repo_id and hf_token_val:
        try:
            handle_hf_readme(
                hf_repo_id=hf_repo_id,
                trigger_word=trigger_word,
                steps=steps,
                learning_rate=learning_rate,
                lora_rank=lora_rank,
            )
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(
                repo_id=hf_repo_id, private=False, exist_ok=True, token=hf_token_val
            )
            api.upload_file(
                path_or_fileobj=str(OUTPUT_DIR / "lora.safetensors"),
                path_in_repo="lora.safetensors",
                repo_id=hf_repo_id,
                token=hf_token_val,
            )
            readme_p = OUTPUT_DIR / "README.md"
            if readme_p.exists():
                api.upload_file(
                    path_or_fileobj=str(readme_p),
                    path_in_repo="README.md",
                    repo_id=hf_repo_id,
                    token=hf_token_val,
                )
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")

    # Remove temporary JOB_DIR if empty
    if JOB_DIR.exists():
        shutil.rmtree(JOB_DIR, ignore_errors=True)

    # Create trained_model.tar containing output files
    with tarfile.open(output_tar, "w") as tar:
        for item in OUTPUT_DIR.iterdir():
            tar.add(item, arcname=item.name)

    return TrainingOutput(weights=Path(output_tar))
