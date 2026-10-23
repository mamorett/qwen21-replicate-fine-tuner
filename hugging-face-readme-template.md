---
license: apache-2.0
license_name: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
language:
- en
tags:
- qwen-image
- diffusers
- lora
- replicate
- text-to-image
base_model: "Qwen/Qwen-Image-2.1"
pipeline_tag: text-to-image
$instance_prompt
---

# $title

<Gallery />

## About this LoRA

This is a [LoRA](https://replicate.com/docs/guides/working-with-loras) for the Qwen-Image 2.1 text-to-image model. It can be used with diffusers or ComfyUI.

It was trained on [Replicate](https://replicate.com/) using AI toolkit.

$trigger_section

## Run this LoRA with an API using Replicate

```py
import replicate

input = {
    "prompt": "$trigger_word",
    "replicate_weights": "https://huggingface.co/$repo_id/resolve/main/lora.safetensors",
}

output = replicate.run("mamorett/qwen21-replicate-fine-tuner", input=input)
for index, item in enumerate(output):
    with open(f"output_{index}.webp", "wb") as file:
        file.write(item.read())
```

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
import torch
from diffusers import QwenImage21Pipeline

pipeline = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")
pipeline.load_lora_weights("$repo_id", weight_name="lora.safetensors")
image = pipeline("$trigger_word").images[0]
image.save("output.png")
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters).

$training_details

## Contribute your own examples

You can use the [community tab](https://huggingface.co/$repo_id/discussions) to add images that show off what you’ve made with this LoRA.
