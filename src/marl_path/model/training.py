"""
Everything related to model training. Contains functions like updating the model weights.
"""

from __future__ import annotations
from typing import Any, List
import torch
import numpy as np
from itertools import groupby

from .utils import build_input_tensor, build_random_input_tensor
from marl_path.shared import Coord


def train_vdn_on_solution(
    model: Any,
    optimizer: Any,
    solution: Any,
    starts: Any,
    goals: Any,
    map: Any,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad()

    num_agents = len(starts)

    values: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for agent_idx in range(num_agents):
        path = [solution[t][agent_idx] for t in range(len(solution))]
        dist_table = (
            model(
                build_input_tensor(map, goals[agent_idx], starts[agent_idx]).to(device)
            )
            .squeeze(0)
            .squeeze(0)
        )
        path = _remove_waiting_from_path(path)
        coord2targets: dict[Coord, int] = _get_targets_for_path(path)

        agent_targets = torch.tensor(
            list(coord2targets.values()), dtype=torch.float32, device=device
        )
        agent_values = _get_via_coordinates(dist_table, list(coord2targets.keys()))

        values.extend(agent_values)
        targets.extend(agent_targets)

    values_tensor = torch.stack(values)
    targets_tensor = torch.stack(targets)

    loss = torch.nn.functional.mse_loss(values_tensor, targets_tensor, reduction="mean")
    loss.backward()
    optimizer.step()
    return loss.item()


def _remove_waiting_from_path(path: List[Coord]) -> List[Coord]:
    return [coord for coord, _ in groupby(path)]


def _get_targets_for_path(path: List[Coord]) -> dict[Coord, int]:
    """Gets the target values for each coordinate in the path. The target value
    for a coordinate is the number of steps until the agent reaches the goal
    from that coordinate, according to the path. If a coordinate appears multiple
    times in the path, the smallest target value is used.

    Args:
        path (List[Coord]): The path of the agent, as a list of coordinates.

    Returns:
        dict[Coord, int]: A dictionary mapping each coordinate in the path to its target value.
    """
    targets = list(range(len(path) - 1, -1, -1))
    coord2targets = dict()
    updated_coords = set()

    # Check for each coordinate what the smallest target value is (for multiple visits) and assign it to the coordinate
    for coord, target in zip(path, targets):
        if coord not in coord2targets:
            coord2targets[coord] = target
        else:
            updated_coords.add(coord)
            coord2targets[coord] = min(coord2targets[coord], target)
    return coord2targets


def _get_via_coordinates(arr: Any, coords: List[Coord]) -> Any:
    """
    Accesses the elements of an 2D numpy array via a list of 2D coordinates in the
    shape [(column, row), ...].

    Args:
        arr (np.ndarray): The 2D numpy array to access.
        coords (List[Coord]): The list of 2D coordinates to access.

    Returns:
        np.ndarray: The elements of the array at the specified coordinates.
    """
    if len(coords) == 0:
        if isinstance(arr, torch.Tensor):
            return torch.tensor([], dtype=arr.dtype, device=arr.device)
        return np.array([], dtype=arr.dtype)
    idx = tuple(np.array(coords).T)
    return arr[idx]


def pretrain_on_default_value(
    model: Any,
    grid: Any,
    optimizer: Any,
    default_value: int | None = None,
    num_epochs: int = 10,
    device: torch.device | None = None,
) -> None:
    """
    Trains the distance-table model, to predict a default value
    for random input. Can be used as a way of initialization.

    Args:
        model (DistanceTableCNN): The model that should be trained
        grid (Grid): The map that the model should be trained on
        default_value (float, optional): The default value that should be predicted. Default is the map size.
        num_epochs (int, optional): How many epochs should be used for training. Defaults to 10.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fill_value: int = (
        grid.shape[0] + grid.shape[1] if default_value is None else default_value
    )

    target_tensor: torch.Tensor = torch.full(
        size=grid.shape, fill_value=fill_value, dtype=torch.float32, device=device
    )

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        random_input: torch.Tensor = build_random_input_tensor(grid, device=device)
        value_tensor = model(random_input).squeeze(0).squeeze(0)
        mean_loss = torch.nn.functional.mse_loss(value_tensor, target_tensor)
        mean_loss.backward()
        optimizer.step()
