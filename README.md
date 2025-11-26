#Video Prediction

This project focuses on video-frame-based prediction with language instruction. The repository currently contains dataset frames and basic annotations to support training and evaluation of vision models.

## Overview
- Task categories: `cover_object`, `drop_object`, `move_object`
- Each sample directory typically includes:
  - Observation frames: `observe_XX.jpg` (e.g., 00–19)
  - Optional target frame: `target.jpg` (for next-frame or key-frame prediction)
  - Annotation: `label.txt` (task-specific labels)
  - Metadata: `video_info.txt` (basic video information)

## Dataset Structure
Data is under `data/final_train_test/train_frames/`, organized by task. Example layout:

data/final_train_test/train_frames/
├── cover_object/
│   └── video_11485/
│       ├── observe_00.jpg
│       ├── ...
│       ├── observe_19.jpg
│       ├── target.jpg
│       └── label.txt
├── drop_object/
│   └── video_97549/
│       ├── observe_00.jpg
│       ├── ...
│       ├── observe_19.jpg
│       ├── video_info.txt
│       └── label.txt
└── move_object/
└── video_11889/
├── observe_00.jpg
├── ...
├── observe_19.jpg
├── video_info.txt
└── label.txt


## Files
- `observe_XX.jpg`: temporally ordered frames; sort by the numeric suffix when loading.
- `target.jpg`: target frame if the task requires predicting the next or a key frame.
- `label.txt`: task label; format varies by task (e.g., class, event time, attributes).
- `video_info.txt`: metadata (e.g., source id, frame rate, sampling policy).

Note: `label.txt` and `video_info.txt` contents may differ across tasks. Inspect a few files first and adapt your parsing logic accordingly.

## Quick Start

### Environment
```bash
python install -e .
```

### Training

```bash
python scripts/train.py # with default argument
```

### Generate Testing Data

```bash
python scripts/generate_result.py --output outputs/test.hdf5
```

## Code Structure


### Model

Under the directory `src/videopred/model`


The model mainly consists of the basic InstructPix2Pix and a MotionEncoder. But the InstructPix2Pix is single frame input.

Given a sequence of $N=20$ consecutive video frames $\{x_1, x_2, \dots, x_{20}\}$ depicting a human–object interaction and a textual instruction $c$ (e.g., "covering a shaving razor with a mirror"), our goal is to predict the future frame $x_{21}$. Instead of using only the last observed frame $x_{20}$ as in conventional image-editing diffusion models, we extract a motion feature vector $\mathbf{m} \in \mathbb{R}^d$ from the entire input sequence using a dedicated motion encoder. This feature $\mathbf{m}$, together with the text embedding $\mathbf{e}_c = \text{CLIP}(c)$  and the reference image $x_{20}$, conditions the diffusion process to generate a temporally coherent prediction of $x_{21}$.

#### MotionEncoder

```python
class MotionEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        # Input: [B, C, T=20, H=96, W=96]
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),   # → H=48, W=48
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),   # → H=24, W=24
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),  # → H=12, W=12
            nn.AdaptiveAvgPool3d((1, 3, 3)),  
            nn.Flatten(),
            nn.Linear(128 * 9, output_dim)    # ← 128 * 3 * 3 = 1152
        )

    def forward(self, x):
        # x: [B, T, C, H, W] → [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        return self.net(x)
```

The MotionEncoder is a CNN with a relatively simple structure, which converts 20 images into the dimension of text embedding.

#### InstructPix2Pix

```python
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
```

The pipeline of InstructPix2Pix is roughly consistent with that in the original paper; the only difference is that we concatenate the output of the motion encoder to the end of the text embedding, which serves as the input to the UNet.