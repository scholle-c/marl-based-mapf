"""
    Contains functions that can be useful for handling the distance table model, such as formatting the input to a correct tensor.
"""
import numpy as np
import torch

def build_input_tensor(grid: np.ndarray, goal: tuple[int, int], start: tuple[int, int]) -> torch.Tensor:
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
