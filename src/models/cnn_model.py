"""
Custom CNN baseline model for Pneumonia detection.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Reusable Conv -> BN -> ReLU -> MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Lightweight custom CNN baseline for Pneumonia detection.

    Architecture: 4 ConvBlocks -> Global Avg Pool -> FC layers
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),          # 224 -> 112
            ConvBlock(32, 64),         # 112 -> 56
            ConvBlock(64, 128),        # 56  -> 28
            ConvBlock(128, 256),       # 28  -> 14
            ConvBlock(256, 512, pool=False),  # 14 -> 14 (no pool)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
