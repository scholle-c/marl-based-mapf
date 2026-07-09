"""Definition of the pathfinding model architecture"""

from __future__ import annotations

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Literal


DELAY_SCALE = 0.1  # scaling of delay so it dosnt dominate the heuristic

OutputActivation = Literal["softplus", "sigmoid"]


class DefaultModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DistanceTableCNN(DefaultModel):
    """
    Plain CNN without pooling that maps 3-channel inputs (map, goal, start)
    to a single-channel distance map.

    output_activation:
        "softplus" (default) — small positive continuous residual, scaled by
            DELAY_SCALE. Used for regression-style delay targets (e.g.
            FirstVisitDelay).
        "sigmoid" — output in [0, 1], used for binary segmentation-style
            targets (e.g. NonOptimalPenaltyDelay). Combine with
            forward_logits() + BCEWithLogitsLoss during training; the
            resulting probability must be scaled by an explicit
            `penalty_scale` hyperparameter before being added to h_bfs
            (see DistTable.compute_delay_model).
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 32,
        depth: int = 4,
        output_activation: OutputActivation = "softplus",
    ):
        super().__init__()
        self.output_activation: OutputActivation = output_activation
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
        (never apply the configured output_activation before that loss)."""
        return self.network(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, C, H, W) -> (batch, 1, H, W), activated."""
        logits = self.forward_logits(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        return torch.nn.functional.softplus(logits) * DELAY_SCALE
