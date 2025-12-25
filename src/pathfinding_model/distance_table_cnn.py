"""CNN architecture for predicting MAPF distance tables from value-map inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


class DistanceTableCNN(nn.Module):
    """
    Plain CNN without pooling that maps 3-channel inputs (map, goal, start)
    to a single-channel distance map.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(depth - 1):
            layers.append(nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            channels = hidden_channels
        layers.append(nn.Conv2d(channels, 1, kernel_size=3, padding=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, 3, H, W) -> (batch, 1, H, W)."""
        return self.network(x)

    def set_default_output_value(self, value: float) -> None:
        """
        Set the bias of the last convolutional layer to a constant value.

        This makes the network output a constant value when the input is zero.
        """
        last_conv: nn.Conv2d = self.network[-1]
        with torch.no_grad():
            last_conv.bias.fill_(value)

    def get(self, start: tuple[int, int], goal: tuple[int, int], grid: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """
        Predict a distance/value table for the provided map, start, and goal.

        Parameters
        ----------
        start:
            Start coordinate as (y, x) in grid coordinates.
        goal:
            Goal coordinate as (y, x) in grid coordinates.
        grid:
            2D map array where non-zero entries denote traversable cells.

        Returns
        -------
        np.ndarray
            Predicted distance table with shape (H, W).
        """
        grid_np = np.asarray(grid)
        if grid_np.ndim != 2:
            raise ValueError("Grid must be a 2D array.")

        start_y, start_x = int(start[0]), int(start[1])
        goal_y, goal_x = int(goal[0]), int(goal[1])

        if not grid_np[goal_y, goal_x]:
            raise ValueError(f"Goal {goal} is not accessible in the provided map.")
        if not grid_np[start_y, start_x]:
            raise ValueError(f"Start {start} is not accessible in the provided map.")

        map_channel = grid_np.astype(np.float32, copy=False)
        goal_channel = np.zeros_like(map_channel, dtype=np.float32)
        goal_channel[goal_y, goal_x] = 1.0
        start_channel = np.zeros_like(map_channel, dtype=np.float32)
        start_channel[start_y, start_x] = 1.0
        stacked = np.stack((map_channel, goal_channel, start_channel), axis=0)

        device = next(self.parameters()).device
        input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = self(input_tensor).squeeze().cpu().numpy()
        return prediction


def load_model(model_path: str | Path, device: str | torch.device | None = None) -> DistanceTableCNN:
    """Load a trained DistanceTableCNN from disk and set it to eval mode."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DistanceTableCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
