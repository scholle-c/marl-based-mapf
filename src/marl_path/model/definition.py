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
    Plain CNN without pooling that maps multi-channel inputs (map, goal,
    start, other agents) to a single-channel [0, 1] delay-probability map.

    Train with forward_logits() + BCEWithLogitsLoss; the resulting
    probability is scaled by an explicit `penalty_scale` hyperparameter
    before being added to h_bfs (see DistTable.compute_delay_model).
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        self._hidden_channels = hidden_channels
        self._depth = depth
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

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-activation network output — feed this to BCEWithLogitsLoss directly
        (never apply sigmoid before that loss)."""
        return self.network(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, C, H, W) -> (batch, 1, H, W) in [0, 1]."""
        return torch.sigmoid(self.forward_logits(x))
