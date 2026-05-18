"""Definition of the pathfinding model architecture"""

from __future__ import annotations

import torch
from torch import nn


class DistanceTableCNN(nn.Module):
    """
    Plain CNN without pooling that maps 3-channel inputs (map, goal, start)
    to a single-channel distance map.
    """

    def __init__(self, in_channels: int = 5, hidden_channels: int = 32, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(depth - 1):
            layers.append(
                nn.Conv2d(
                    channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                )
            )
            layers.append(nn.LeakyReLU(inplace=True))
            channels = hidden_channels
        layers.append(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, padding_mode="replicate")
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, 3, H, W) -> (batch, 1, H, W)."""
        return self.network(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 5, features: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(
            in_channels, features, kernel_size=3, padding=1, padding_mode="replicate"
        )  # TODO UNet testen
        self.enc2 = nn.Conv2d(
            features,
            features * 2,
            kernel_size=3,
            padding=1,
            stride=2,
            padding_mode="replicate",
        )

        # Bottleneck
        self.bottleneck = nn.Conv2d(
            features * 2,
            features * 2,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.dec2 = nn.Conv2d(
            features * 2, features, kernel_size=3, padding=1, padding_mode="replicate"
        )

        self.final = nn.Conv2d(
            features, 1, kernel_size=3, padding=1, padding_mode="replicate"
        )
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        # Encoder
        s1 = self.relu(self.enc1(x))
        s2 = self.relu(self.enc2(s1))

        # Bottleneck
        b = self.relu(self.bottleneck(s2))

        # Decoder
        up2 = self.upconv2(b)
        merge2 = torch.cat([up2, s1], dim=1)
        d2 = self.relu(self.dec2(merge2))

        return self.final(d2)
