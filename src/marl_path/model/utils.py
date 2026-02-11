"""
Contains helper functions for the model package.
"""

import numpy as np
import torch
from marl_path.shared import Grid


def build_input_tensor(
    grid: Grid,
    goal: tuple[int, int],
    start: tuple[int, int],
    device: torch.device | None = None,
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
    tensor = torch.from_numpy(stacked).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    tensor = _add_coords(tensor, goal)
    return tensor


def build_random_input_tensor(
    grid: Grid,
    device: torch.device | None = None,
    use_random_input_channels: bool = True,
) -> torch.Tensor:
    random_goal = (0, 0)
    if use_random_input_channels:
        map_channel = np.random.random(grid.shape)
        goal_channel = np.random.random(grid.shape)
        start_channel = np.random.random(grid.shape)
    else:
        map_channel = grid.astype(np.float32, copy=False)
        goal_channel = np.zeros_like(map_channel, dtype=np.float32)
        start_channel = np.zeros_like(map_channel, dtype=np.float32)

        # Set random goal and start positions
        accessible_positions = np.argwhere(grid)
        if len(accessible_positions) < 2:
            raise ValueError(
                "Grid must have at least two accessible positions for random start and goal."
            )
        random_indices = np.random.choice(
            len(accessible_positions), size=2, replace=False
        )
        random_goal = tuple(accessible_positions[random_indices[0]])
        random_start = tuple(accessible_positions[random_indices[1]])
        goal_channel[random_goal] = 1.0
        start_channel[random_start] = 1.0

    stacked = np.stack(
        (map_channel, goal_channel, start_channel), axis=0, dtype=np.float32
    )
    tensor = torch.from_numpy(stacked).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    tensor = _add_coords(tensor, random_goal)
    return tensor


def _add_coords(
    input_tensor: torch.Tensor, goal: tuple[int, int], use_relative_coords: bool = True
) -> torch.Tensor:
    """Add coordinate channels to the input tensor.

    Args:
        input_tensor: Input tensor of shape (batch, channels, height, width).

    Returns:
        Tensor of shape (batch, channels + 2, height, width) with added coordinate channels.
    """
    batch_size, _, height, width = input_tensor.shape

    if use_relative_coords:
        target_y, target_x = goal
        y_range = torch.arange(height, dtype=torch.float32)
        x_range = torch.arange(width, dtype=torch.float32)

        y_rel = (y_range - target_y) / height
        x_rel = (x_range - target_x) / width

        y_coords = y_rel.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        x_coords = x_rel.view(1, 1, 1, width).expand(batch_size, 1, height, width)
    else:
        y_coords = (
            torch.linspace(0, 1, steps=height)
            .view(1, 1, height, 1)
            .expand(batch_size, 1, height, width)
        )
        x_coords = (
            torch.linspace(0, 1, steps=width)
            .view(1, 1, 1, width)
            .expand(batch_size, 1, height, width)
        )

    if input_tensor.is_cuda:
        y_coords = y_coords.cuda()
        x_coords = x_coords.cuda()

    return torch.cat([input_tensor, y_coords, x_coords], dim=1)
