# scripts/train.py

import argparse
import logging
import math
import os
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from videopred.model.wrap_model import WrapInstructPix2Pix
from videopred.model.motion_encoder import MotionEncoder
from videopred.dataset import VideoFramesDataset


    
def parse_args():
    parser = argparse.ArgumentParser(description="Train Video-conditioned InstructPix2Pix with LoRA")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--data_root", type=str, default="data/final_train_test/train_frames")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA for UNet")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = get_logger(__name__)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    # ============================
    # Initialize model
    # ============================
    model = WrapInstructPix2Pix(
        pretrained_model_name_or_path="model/instruct-pix2pix",
        motion_output_dim=768,
        device=accelerator.device,
        freeze_vae=True,
        freeze_text_encoder=True,
    )

    # ============================
    # Apply LoRA to UNet if requested
    # ============================
    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        model.unet = get_peft_model(model.unet, lora_config)
        model.unet.print_trainable_parameters()
        logger.info("✅ LoRA applied to UNet")

    # ============================
    # Optimizer & Scheduler
    # ============================
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # ============================
    # Dataset & Dataloader
    # ============================
    dataset = VideoFramesDataset(
        root=args.data_root,
        sub_dir = ["cover_object", "drop_object", "move_object"],
        num_frames=args.num_frames,
        resolution=args.resolution,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    # ============================
    # Prepare with Accelerate
    # ============================
    model.unet, model.motion_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model.unet, model.motion_encoder, optimizer, dataloader, lr_scheduler
    )

    # Keep other components on device but not prepared (they're frozen)
    model.vae.to(accelerator.device)
    model.text_encoder.to(accelerator.device)

    # ============================
    # Training loop
    # ============================
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(args.resume_from_checkpoint.split("-")[-1])
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        model.unet.train()
        model.motion_encoder.train()

        for batch in dataloader:
            with accelerator.accumulate(model.unet, model.motion_encoder):
                video = batch["video"].to(accelerator.device, dtype=torch.float32)  # [B, T, C, H, W]
                target = batch["target"].to(accelerator.device, dtype=torch.float32)  # [B, C, H, W]
                prompts = batch["prompt"]

                # Encode target image to latent
                with torch.no_grad():
                    original_latents = model.encode_image(video[:, -1])  # [B, 4, H//8, W//8]
                    target_latent = model.encode_image(target)  # [B, 4, H//8, W//8]

                # Sample noise and timesteps
                noise = torch.randn_like(target_latent)
                bsz = target_latent.shape[0]
                timesteps = torch.randint(
                    0, model.scheduler.config.num_train_timesteps, (bsz,), device=target_latent.device
                ).long()

                noisy_target_latents = model.scheduler.add_noise(target_latent, noise, timesteps)

                # Prepare condition: text + motion
                encoder_hidden_states = model.prepare_condition(prompts, video)

                # Predict noise
                model_pred = model.forward_unet(original_latents, noisy_target_latents, timesteps, encoder_hidden_states)

                loss = F.mse_loss(model_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

    # ============================
    # Save final model
    # ============================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrap_unet = accelerator.unwrap_model(model.unet)
        unwrap_motion = accelerator.unwrap_model(model.motion_encoder)

        output_dir = Path(args.output_dir) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.use_lora:
            unwrap_unet.save_pretrained(output_dir / "unet_lora")
        else:
            unwrap_unet.save_pretrained(output_dir / "unet")

        torch.save(unwrap_motion.state_dict(), output_dir / "motion_encoder.pth")
        logger.info(f"Final model saved to {output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()