"""
Optimized LoRA training for 4GB VRAM.
Pre-computes VAE latents and text embeddings so the training loop
only runs UNet forward/backward (~1-2s/step instead of ~17s).
"""
import os
import gc
import torch
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from config import OUTPUT_DIR, SD_MODEL, LORA_PATH, LORA_SCALE, TRIGGER_WORD

TRAINING_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_images")
RESOLUTION = 512
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
LR_SCHEDULER = "constant"
LR_WARMUP_STEPS = 50
MAX_TRAIN_STEPS = 1000
RANK = 32
SEED = 42
MIXED_PRECISION = "fp16"
SAVE_EVERY = 500


class PrecomputedDataset(Dataset):
    def __init__(self, latents, embeddings):
        self.latents = latents
        self.embeddings = embeddings

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {"latent": self.latents[idx], "encoder_hidden_states": self.embeddings[idx]}


def main():
    parser = argparse.ArgumentParser(description="Train a Style LoRA for Stable Diffusion")
    parser.add_argument("--image_dir", default=TRAINING_IMAGES_DIR)
    parser.add_argument("--output", default=LORA_PATH)
    parser.add_argument("--trigger_word", default=TRIGGER_WORD)
    parser.add_argument("--steps", type=int, default=MAX_TRAIN_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
    )
    set_seed(SEED)

    device = accelerator.device
    dtype = torch.float16 if MIXED_PRECISION == "fp16" else torch.float32

    image_paths = sorted([
        os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if accelerator.is_main_process:
        print(f"Training images: {len(image_paths)}")
        print(f"Base model: {SD_MODEL}")
        print(f"Trigger word: '{args.trigger_word}'")
        print(f"LoRA rank: {args.rank} | Steps: {args.steps} | LR: {args.lr}")
        print(f"Resolution: {args.resolution} | Batch: {args.batch_size} | Accum: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"Output: {args.output}\n")

    print("[1/4] Loading base model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    del pipe
    gc.collect()

    print("[2/4] Pre-computing VAE latents and text embeddings...")
    transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    latents_list = []
    embeddings_list = []
    text_inputs = tokenizer(
        [args.trigger_word],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0].detach().cpu()

    for img_path in tqdm(image_paths, desc="Encoding images", disable=not accelerator.is_local_main_process):
        image = Image.open(img_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        latents_list.append(latents.detach().cpu())
        embeddings_list.append(text_embeddings)

    vae.to("cpu")
    text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    all_latents = torch.cat(latents_list, dim=0)
    all_embeddings = torch.cat(embeddings_list, dim=0)
    del latents_list, embeddings_list
    gc.collect()

    dataset = PrecomputedDataset(all_latents, all_embeddings)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print(f"[3/4] Setting up LoRA (rank={args.rank})...")
    unet.requires_grad_(False)
    unet.to(device, dtype=dtype)
    unet.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "conv1", "conv2"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()
    cast_training_params(unet, dtype=torch.float32)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} ({trainable/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-2)
    lr_scheduler = get_scheduler(
        LR_SCHEDULER, optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS * accelerator.num_processes,
        num_training_steps=args.steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    print(f"[4/4] Training ({args.steps} steps)...")
    progress_bar = tqdm(range(args.steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    global_step = 0
    while global_step < args.steps:
        for batch in train_dataloader:
            if global_step >= args.steps:
                break

            with accelerator.accumulate(unet):
                latents = batch["latent"].to(device, dtype=dtype)
                encoder_hidden_states = batch["encoder_hidden_states"].to(device, dtype=dtype)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % max(1, args.steps // 20) == 0:
                    progress_bar.set_postfix(loss=f"{loss.detach().item():.4f}")

                if global_step % SAVE_EVERY == 0 and accelerator.is_main_process:
                    save_path = args.output.replace(".safetensors", f"_step{global_step}.safetensors")
                    accelerator.unwrap_model(unet).save_pretrained(
                        os.path.splitext(save_path)[0], safe_serialization=True
                    )
                    adapter_dir = os.path.splitext(save_path)[0]
                    adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
                    if os.path.exists(adapter_file) and adapter_file != save_path:
                        import shutil
                        os.rename(adapter_file, save_path)
                        shutil.rmtree(adapter_dir)
                    print(f"\nCheckpoint: {save_path}")

        if global_step >= args.steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.splitext(args.output)[0]
        accelerator.unwrap_model(unet).save_pretrained(final_dir, safe_serialization=True)
        adapter_file = os.path.join(final_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            if os.path.exists(args.output):
                os.remove(args.output)
            os.rename(adapter_file, args.output)
            import shutil
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)

        if os.path.exists(args.output):
            size_mb = os.path.getsize(args.output) / 1024 / 1024
            print(f"\nDone! LoRA saved: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
