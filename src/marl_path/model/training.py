"""
Everything related to model training. Contains functions like updating the model weights.
"""

from __future__ import annotations
from typing import Any, List
import torch
import numpy as np

from marl_path.shared import Coord
from .feature_extraction import FeatureExtractor, BasicExtractor


def train_vdn_on_solution(
    model: Any,
    optimizer: Any,
    solution: Any,
    starts: Any,
    goals: Any,
    map: Any,
    device: torch.device,
    extractor: FeatureExtractor,
) -> float:
    model.train()
    optimizer.zero_grad()

    num_agents = len(starts)

    target_q_tot: torch.Tensor = torch.zeros(
        len(solution), dtype=torch.float32, device=device
    )
    values_q_tot = None

    # TODO: Maybe this can be parallized or done using np methods on solution?
    for agent_idx in range(num_agents):
        path = [solution[t][agent_idx] for t in range(len(solution))]
        other_agent_goals = [goals[i] for i in range(num_agents) if i != agent_idx]
        dist_table = (
            model(
                extractor.extract(
                    map,
                    goals[agent_idx],
                    starts[agent_idx],
                    other_agent_goals,
                    device=device,
                )
            )
            .squeeze(0)
            .squeeze(0)
        )

        agent_values = _get_via_coordinates(dist_table, path)
        agent_targets = _get_path_target(path)

        target_q_tot += torch.tensor(agent_targets, dtype=torch.float32, device=device)
        if values_q_tot is None:
            values_q_tot = agent_values
        else:
            values_q_tot += agent_values

    assert values_q_tot is not None
    loss = torch.nn.functional.mse_loss(values_q_tot, target_q_tot, reduction="mean")
    loss.backward()
    optimizer.step()
    return loss.item()


def _get_path_target(path: Any) -> List[int]:
    """
    Determines the distance values for a given path of length > 0, where the agent
    has reached its goal. Works like this:
    1.) Iterate from goal to start, start with trgt = 0
    2.) When the agent moved, add trgt++ to list of targets
    3.) When the agent waited, add trgt to list of targets

    Args:
        path (Any): Path of the agent. len(path) must be greater than zero.

    Returns:
        List[int]: A list with distance-target values
    """
    targets = [0]
    trgt: int = 0
    coord_prev = path[-1]
    goal = path[-1]
    for coord in reversed(path[:-1]):
        if coord != coord_prev:
            trgt += 1

        if coord == goal:
            targets.append(0)
        else:
            targets.append(trgt)

        coord_prev = coord
    targets.reverse()
    return targets


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
    extractor: FeatureExtractor | None = None,
) -> None:
    """
    Trains the distance-table model, to predict a default value
    for random input. Can be used as a way of initialization.

    Args:
        model (DistanceTableCNN): The model that should be trained
        grid (Grid): The map that the model should be trained on
        default_value (float, optional): The default value that should be predicted. Default is the map size.
        num_epochs (int, optional): How many epochs should be used for training. Defaults to 10.
        extractor: Feature extractor to use. Defaults to BasicExtractor.
    """
    if extractor is None:
        extractor = BasicExtractor()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fill_value: int = (
        grid.shape[0] + grid.shape[1] if default_value is None else default_value
    )

    target_tensor: torch.Tensor = torch.full(
        size=grid.shape, fill_value=fill_value, dtype=torch.float32, device=device
    )

    accessible = np.argwhere(grid)
    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        idx = np.random.choice(len(accessible), size=2, replace=False)
        goal: tuple[int, int] = (int(accessible[idx[0], 0]), int(accessible[idx[0], 1]))
        start: tuple[int, int] = (
            int(accessible[idx[1], 0]),
            int(accessible[idx[1], 1]),
        )
        random_input: torch.Tensor = extractor.extract(
            grid, goal, start, [], device=device
        )
        value_tensor = model(random_input).squeeze(0).squeeze(0)
        mean_loss = torch.nn.functional.mse_loss(value_tensor, target_tensor)
        mean_loss.backward()
        optimizer.step()
