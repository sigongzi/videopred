from typing import Optional, Union, List
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from videopred.model.motion_encoder import MotionEncoder  # 假设 MotionEncoder 在同目录下
import os

class WrapInstructPix2Pix(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "timbrooks/instruct-pix2pix",
        motion_output_dim: int = 768,
        checkpoint_dir: str = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        freeze_vae: bool = True,
        freeze_text_encoder: bool = True,
        enable_xformers: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        print(pretrained_model_name_or_path)

        # === Load base components from InstructPix2Pix ===
        print(f"Loading InstructPix2Pix from {pretrained_model_name_or_path}...")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        if checkpoint_dir is not None:
            self.unet = UNet2DConditionModel.from_pretrained(checkpoint_dir, subfolder="unet")
        else:
            self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

        self.scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")


        # === Motion Encoder ===
        self.motion_encoder = MotionEncoder(output_dim=motion_output_dim)
        if checkpoint_dir is not None:
            motion_encoder_weights_path = os.path.join(checkpoint_dir, "motion_encoder.pth")
            state_dict = torch.load(motion_encoder_weights_path, map_location="cpu")
            self.motion_encoder.load_state_dict(state_dict)
            print(f"Loaded motion encoder weights from {motion_encoder_weights_path}")

        # === Move to device ===
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
        self.motion_encoder.to(self.device)

        # === Freeze components if needed ===
        if freeze_vae:
            self.vae.requires_grad_(False)
        if freeze_text_encoder:
            self.text_encoder.requires_grad_(False)

        # === Optional: Enable xFormers for memory efficiency ===
        if enable_xformers and hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
            self.unet.enable_xformers_memory_efficient_attention()

        # Set eval mode for frozen parts (train script can override)
        self.vae.eval()
        self.text_encoder.eval()

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.
        Args:
            image: Tensor of shape [B, C, H, W] in range [-1, 1]
        Returns:
            latent: [B, 4, H//8, W//8]
        """
        posterior = self.vae.encode(image).latent_dist
        latent = posterior.sample() * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.
        """
        image = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        return image.clamp(-1, 1)

    @torch.no_grad()
    def encode_text(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text prompts to embeddings.
        Returns: [B, 77, 768]
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        return self.text_encoder(tokens).last_hidden_state

    @torch.no_grad()
    def encode_motion(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of video clips to motion features.
        Args:
            video_frames: [B, T, C, H, W] with values in [-1, 1]
        Returns:
            motion_feat: [B, motion_output_dim]
        """
        return self.motion_encoder(video_frames)

    def prepare_condition(
        self,
        prompts: Union[str, List[str]],
        video_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare combined text + motion condition for UNet.
        Returns:
            encoder_hidden_states: [B, 78, 768]  (77 text tokens + 1 motion token)
        """
        text_embeds = self.encode_text(prompts)  # [B, 77, D]
        motion_feat = self.encode_motion(video_frames)  # [B, D]
        motion_token = motion_feat.unsqueeze(1)  # [B, 1, D]
        return torch.cat([text_embeds, motion_token], dim=1)  # [B, 78, D]

    def forward_unet(
        self,
        original_latents: torch.Tensor,
        noisy_target_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        
    ) -> torch.Tensor:
        """
        Forward pass through UNet.
        Args:
            original_latents: [B, 4, H, W] - original image latents 
            noisy_target_latents: [B, 4, H, W] - noisy version of the target image latents
            timesteps: [B] - diffusion timesteps
            encoder_hidden_states: [B, 78, 768] - text + motion embeddings
        """

        # Concatenate condition latents and noisy latents along channel dimension
        # This is typical for InstructPix2Pix-like models
        model_input = torch.cat([original_latents, noisy_target_latents], dim=1)  # [B, 8, H, W]

        return self.unet(
            model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    def get_trainable_parameters(self):
        """Return parameters to optimize (e.g., for optimizer)."""
        params = []
        params.extend(self.unet.parameters())
        params.extend(self.motion_encoder.parameters())
        return params

    @torch.no_grad()
    def predict(
        self,
        video_frames: torch.Tensor,
        prompts: Union[str, List[str]],
        num_inference_steps: int = 100,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate a single output image conditioned on a video clip and text prompt.

        Args:
            video_frames: Tensor of shape [B, T, C, H, W], values in [-1, 1]
            prompts: Text prompt(s), length B
            num_inference_steps: Number of denoising steps (e.g., 50)
            guidance_scale: Classifier-free guidance scale (set to 0 or 1 to disable)
            generator: Random generator for reproducibility

        Returns:
            predicted_image: [B, C, H, W] in range [-1, 1]
        """
        self.unet.eval()
        self.motion_encoder.eval()

        device = self.device
        dtype = self.unet.dtype

        batch_size = video_frames.shape[0]

        # === Encode the "original" image (last frame of video) ===
        original_image = video_frames[:, -1]  # [B, C, H, W]
        original_latents = self.encode_image(original_image)  # [B, 4, H//8, W//8]

        # === Prepare encoder hidden states (text + motion) ===
        encoder_hidden_states = self.prepare_condition(prompts, video_frames)  # [B, 78, 768]

        # === Handle classifier-free guidance ===
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            # Encode empty prompt
            uncond_prompts = [""] * batch_size
            uncond_video = torch.zeros_like(video_frames)  # 或者也可以用原 video？但通常 motion 也置零？
            # 更合理做法：motion 保留？但标准做法是全空条件。这里我们对 motion 也清零以实现完全无条件。
            uncond_motion = torch.zeros_like(video_frames)
            uncond_encoder_hidden_states = self.prepare_condition(uncond_prompts, uncond_motion)
            # Concatenate
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states])

        # === Initialize random noise ===
        height, width = original_latents.shape[-2:]
        latents = torch.randn(
            (batch_size, 4, height, width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # === Set scheduler timesteps ===
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        latents = latents * scheduler.init_noise_sigma

        # === Denoising loop ===
        for t in scheduler.timesteps:
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            original_latents_input = torch.cat([original_latents] * 2) if do_classifier_free_guidance else original_latents

            # Concatenate with original image latents (InstructPix2Pix style)
            model_input = torch.cat([original_latents_input, latent_model_input], dim=1)  # [B(×2), 8, H, W]

            # Predict noise
            noise_pred = self.unet(
                model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # === Decode to image ===
        predicted_image = self.decode_latent(latents)  # [B, C, H, W], [-1, 1]
        return predicted_image