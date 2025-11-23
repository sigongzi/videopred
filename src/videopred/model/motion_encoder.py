# src/videopred/motion_encoder.py

import torch
import torch.nn as nn

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
            nn.AdaptiveAvgPool3d((1, 3, 3)),  # ← 改为 (1, 3, 3) 更适合 12x12
            nn.Flatten(),
            nn.Linear(128 * 9, output_dim)    # ← 128 * 3 * 3 = 1152
        )

    def forward(self, x):
        # x: [B, T, C, H, W] → [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        return self.net(x)