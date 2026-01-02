"""
Is used when you want to use a trained model for inference. Contains functions
to load a model and run a forward pass.
"""

from __future__ import annotations

from .definition import DistanceTableCNN
import torch
import numpy as np
from typing import Any


def load_model(model_path: str, device: str | None = None) -> Any:
    """
    Load a trained model for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistanceTableCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_distance_table(
    model: DistanceTableCNN, grid: Any, start: tuple[int, int], goal: tuple[int, int]
) -> Any:
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

    device = next(model.parameters()).device
    input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor).squeeze().cpu().numpy()
    return prediction
