"""
Contains helper functions for the model package.
"""

import numpy as np
import torch
from marl_path.shared import Grid


def build_input_tensor(
    grid: Grid, goal: tuple[int, int], start: tuple[int, int]
) -> torch.Tensor:
    if grid.ndim != 2:
        raise ValueError("Grid must be a 2D array.")
    if not grid[goal]:
        raise ValueError(f"Goal {goal} is not accessible in the provided map.")
    if not grid[start]:
        raise ValueError(f"Start {start} is not accessible in the provided map.")

    map_channel = grid.astype(np.float32, copy=False)
    goal_channel = np.zeros_like(map_channel, dtype=np.float32)
    goal_channel[goal] = 1.0
    start_channel = np.zeros_like(map_channel, dtype=np.float32)
    start_channel[start] = 1.0

    stacked = np.stack((map_channel, goal_channel, start_channel), axis=0)
    return torch.from_numpy(stacked).unsqueeze(0)


def build_random_input_tensor(grid: Grid) -> torch.Tensor:
    map_channel = np.random.random(grid.shape)
    goal_channel = np.random.random(grid.shape)
    start_channel = np.random.random(grid.shape)

    stacked = np.stack(
        (map_channel, goal_channel, start_channel), axis=0, dtype=np.float32
    )
    return torch.from_numpy(stacked).unsqueeze(0)
