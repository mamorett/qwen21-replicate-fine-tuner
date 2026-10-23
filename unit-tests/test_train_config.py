import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from train import build_train_config, extract_zip, parse_resolution


def test_submodule_pin_guard():
    """Test 1: Submodule Pin Guard.
    Assert ai-toolkit is locked at 14f8f95090689b585093e5aaf48bc1e2e0a67ea4.
    """
    repo_root = Path(__file__).parent.parent
    ai_toolkit_path = repo_root / "ai-toolkit"

    # Try git rev-parse inside ai-toolkit first
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(ai_toolkit_path),
            )
            .decode()
            .strip()
        )
    except Exception:
        # Fallback to git submodule status in repo_root
        status = (
            subprocess.check_output(
                ["git", "submodule", "status"],
                cwd=str(repo_root),
            )
            .decode()
            .strip()
        )
        sha = None
        for line in status.splitlines():
            if "ai-toolkit" in line:
                sha = line.split()[0].lstrip("+-U")
                break

    assert sha == "14f8f95090689b585093e5aaf48bc1e2e0a67ea4", (
        f"Expected ai-toolkit commit 14f8f95090689b585093e5aaf48bc1e2e0a67ea4, got {sha}"
    )


def test_template_parity():
    """Test 2: Template Parity.
    Call build_train_config() with default arguments and assert deep equality
    with config/qwen_image_21_lora.yaml.
    """
    repo_root = Path(__file__).parent.parent
    ref_yaml_path = repo_root / "config" / "qwen_image_21_lora.yaml"
    assert ref_yaml_path.exists(), (
        "Reference YAML config/qwen_image_21_lora.yaml missing!"
    )

    with ref_yaml_path.open(encoding="utf-8") as f:
        reference_data = yaml.safe_load(f)

    generated_config = build_train_config()
    # Normalize via json roundtrip to dict
    generated_data = json.loads(json.dumps(generated_config))

    assert generated_data == reference_data, (
        "Generated config does not match reference YAML template!"
    )

    # Invariant assertions
    process = generated_data["config"]["process"][0]
    model = process["model"]
    train = process["train"]

    assert model["arch"] == "qwen_image_2"
    assert model["low_vram"] is False
    assert model["quantize"] is False
    assert model["quantize_te"] is False
    assert train["dtype"] == "bf16"
    assert train["optimizer"] == "adamw8bit"
    assert train["lr"] == 0.0001
    assert train["steps"] == 2000


def test_dynamic_overrides():
    """Test 3: Dynamic Overrides.
    Verify that lora_rank=32 sets both linear=32 and linear_alpha=32,
    rgba=True sets model_kwargs.rgba=True, and low_vram=True toggles model.low_vram=True.
    """
    cfg = build_train_config(lora_rank=32, rgba=True, low_vram=True)
    process = cfg["config"]["process"][0]

    assert process["network"]["linear"] == 32
    assert process["network"]["linear_alpha"] == 32
    assert process["model"]["model_kwargs"]["rgba"] is True
    assert process["model"]["low_vram"] is True


def test_wandb_sampling_toggle():
    """Test 4: W&B Sampling Toggle.
    Verify that disable_sampling stays True by default, and switches to False with prompts
    populated only when both wandb_api_key and wandb_sample_prompts are provided.
    """
    # Defaults
    default_cfg = build_train_config()
    default_proc = default_cfg["config"]["process"][0]
    assert default_proc["train"]["disable_sampling"] is True
    assert default_proc["sample"]["prompts"] == []

    # Only API key, no prompts
    key_only_cfg = build_train_config(wandb_api_key="secret-key")
    key_only_proc = key_only_cfg["config"]["process"][0]
    assert key_only_proc["train"]["disable_sampling"] is True
    assert key_only_proc["sample"]["prompts"] == []

    # Both API key and sample prompts
    prompts_text = "photo of a magical forest\na futuristic city at night"
    active_cfg = build_train_config(
        wandb_api_key="secret-key",
        wandb_sample_prompts=prompts_text,
        wandb_sample_interval=250,
    )
    active_proc = active_cfg["config"]["process"][0]
    assert active_proc["train"]["disable_sampling"] is False
    assert active_proc["sample"]["prompts"] == [
        "photo of a magical forest",
        "a futuristic city at night",
    ]
    assert active_proc["sample"]["sample_every"] == 250


def test_resolution_validator():
    """Test 5: Resolution Validator.
    Verify "1024, 1536" parses to [1024, 1536] and invalid non-multiples of 16 raise ValueError.
    """
    assert parse_resolution("1024, 1536") == [1024, 1536]
    assert parse_resolution("512, 1024, 2048") == [512, 1024, 2048]

    # Non-multiple of 16
    with pytest.raises(ValueError, match="divisible by 16"):
        parse_resolution("1000")

    with pytest.raises(ValueError, match="divisible by 16"):
        parse_resolution("1024, 1005")

    # Out of range (< 512 or > 2048)
    with pytest.raises(ValueError, match="between 512 and 2048"):
        parse_resolution("256")

    with pytest.raises(ValueError, match="between 512 and 2048"):
        parse_resolution("4096")


def test_archive_extraction():
    """Test 6: Archive Extraction.
    Verify __MACOSX and ._* resource fork files are ignored when extracting input_images.zip.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_path = temp_path / "test_input.zip"
        dest_path = temp_path / "extracted"

        # Create test zip with valid files and mac resource fork artifacts
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("photo1.jpg", b"fake image bytes 1")
            zf.writestr("photo2.png", b"fake image bytes 2")
            zf.writestr("photo2.txt", b"a caption for photo 2")
            zf.writestr("__MACOSX/._photo1.jpg", b"mac os x metadata")
            zf.writestr("subfolder/__MACOSX/._photo2.png", b"mac os x metadata")
            zf.writestr("._photo1.jpg", b"mac os x metadata")

        extract_zip(zip_path, dest_path)

        extracted_files = [f.name for f in dest_path.iterdir()]
        assert "photo1.jpg" in extracted_files
        assert "photo2.png" in extracted_files
        assert "photo2.txt" in extracted_files

        # Ensure no resource forks or MACOSX files were extracted
        for f in dest_path.rglob("*"):
            assert not f.name.startswith("._"), f"Resource fork extracted: {f}"
            assert "__MACOSX" not in str(f), f"MACOSX extracted: {f}"
