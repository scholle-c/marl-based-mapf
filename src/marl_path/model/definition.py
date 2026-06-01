"""Definition of the pathfinding model architecture"""

from __future__ import annotations

import torch
from torch import nn
from abc import ABC, abstractmethod


class DefaultModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DistanceTableCNN(DefaultModel):
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
