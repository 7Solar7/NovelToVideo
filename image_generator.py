import os
import torch
from diffusers import StableDiffusionPipeline
from config import (
    SD_MODEL, SD_WIDTH, SD_HEIGHT, FIXED_SEED, NUM_INFERENCE_STEPS,
    OUTPUT_DIR, NEGATIVE_PROMPT, LORA_PATH, LORA_SCALE
)


def load_pipeline():
    print(f"Loading Stable Diffusion model: {SD_MODEL}")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA weights from: {LORA_PATH}")
        pipe.load_lora_weights(LORA_PATH)
        pipe.fuse_lora(lora_scale=LORA_SCALE)

    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        pipe.enable_attention_slicing()

    return pipe


def generate_images(segments, pipe=None):
    if pipe is None:
        pipe = load_pipeline()

    generator = None
    if FIXED_SEED is not None:
        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(FIXED_SEED)

    image_paths = []
    for seg in segments:
        idx = seg["index"]
        prompt = seg["prompt"]
        output_path = os.path.join(OUTPUT_DIR, f"img_{idx:03d}.png")

        if os.path.exists(output_path):
            print(f"Image {idx} already exists, skipping (cached): {output_path}")
            image_paths.append(output_path)
            continue

        print(f"Generating image {idx}/{len(segments)}: {prompt[:60]}...")
        image = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=SD_WIDTH,
            height=SD_HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
        ).images[0]

        image.save(output_path)
        print(f"Saved: {output_path}")
        image_paths.append(output_path)

    return image_paths
