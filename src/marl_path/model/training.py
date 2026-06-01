"""
Everything related to model training. Contains functions like updating the model weights.
"""

from __future__ import annotations
from collections import deque
from typing import Any, List, Tuple
import torch
import numpy as np

from marl_path.shared import Coord, get_neighbors
from .feature_extraction import FeatureExtractor, BasicExtractor


def compute_vdn_tensors(
    model: Any,
    solution: Any,
    starts: Any,
    goals: Any,
    map: Any,
    device: torch.device,
    extractor: FeatureExtractor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_agents = len(starts)
    dist_tables = _batch_dist_tables(model, extractor, map, starts, goals, device)

    target_q_tot: torch.Tensor = torch.zeros(
        len(solution), dtype=torch.float32, device=device
    )
    values_q_tot = None

    for agent_idx in range(num_agents):
        path = [solution[t][agent_idx] for t in range(len(solution))]
        agent_values = _get_via_coordinates(dist_tables[agent_idx], path)
        agent_targets = _get_path_target(path)

        target_q_tot += torch.tensor(agent_targets, dtype=torch.float32, device=device)
        if values_q_tot is None:
            values_q_tot = agent_values
        else:
            values_q_tot += agent_values

    assert values_q_tot is not None
    return values_q_tot, target_q_tot


def compute_individual_tensors(
    model: Any,
    solution: Any,
    starts: Any,
    goals: Any,
    map: Any,
    device: torch.device,
    extractor: FeatureExtractor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like compute_vdn_tensors but without VDN decomposition: each agent's
    predicted values and targets are concatenated rather than summed, so the
    loss is computed independently per agent per timestep."""
    num_agents = len(starts)
    dist_tables = _batch_dist_tables(model, extractor, map, starts, goals, device)

    all_values: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for agent_idx in range(num_agents):
        path = [solution[t][agent_idx] for t in range(len(solution))]
        all_values.append(_get_via_coordinates(dist_tables[agent_idx], path))
        all_targets.append(
            torch.tensor(_get_path_target(path), dtype=torch.float32, device=device)
        )

    return torch.cat(all_values), torch.cat(all_targets)


def _batch_dist_tables(
    model: Any,
    extractor: FeatureExtractor,
    map: Any,
    starts: Any,
    goals: Any,
    device: torch.device,
) -> torch.Tensor:
    """Single batched forward pass for all agents, returns (num_agents, H, W)."""
    num_agents = len(starts)
    inputs = torch.cat(
        [
            extractor.extract(
                map,
                goals[i],
                starts[i],
                [goals[j] for j in range(num_agents) if j != i],
                device=device,
            )
            for i in range(num_agents)
        ]
    )  # (num_agents, C, H, W)
    return model(inputs).squeeze(1)  # (num_agents, H, W)


def update_from_batch(
    model: Any,
    optimizer: Any,
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    loss_fn=torch.nn.functional.mse_loss,
) -> float:
    if not batch:
        return float("nan")
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for values, targets in batch:
        loss = loss_fn(values, targets)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss / len(batch)


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


def _compute_bfs_table(grid: Any, goal: Coord) -> np.ndarray:
    """Full BFS distance table from goal. Unreachable/wall cells keep NIL = grid.size."""
    NIL = grid.size
    table = np.full(grid.shape, NIL, dtype=np.float32)
    table[goal] = 0
    Q: deque[Coord] = deque([goal])
    while Q:
        u = Q.popleft()
        d = int(table[u])
        for v in get_neighbors(grid, u):
            if d + 1 < table[v]:
                table[v] = d + 1
                Q.append(v)
    return table


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


def pretrain_on_bfs(
    model: Any,
    grid: Any,
    optimizer: Any,
    num_epochs: int = 10,
    device: torch.device | None = None,
    extractor: FeatureExtractor | None = None,
) -> None:
    """
    Trains the distance-table model, to predict the distance to a random goal
    from a random start on the given grid, using BFS as the target heuristic.

    Args:
        model (DistanceTableCNN): The model that should be trained
        grid (Grid): The map that the model should be trained on
        num_epochs (int, optional): How many epochs should be used for training. Defaults to 10.
        extractor: Feature extractor to use. Defaults to BasicExtractor.
    """
    if extractor is None:
        extractor = BasicExtractor()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        bfs_table = _compute_bfs_table(grid, goal)
        target_tensor = torch.tensor(bfs_table, dtype=torch.float32, device=device)

        # Mask out walls so their NIL values don't dominate the loss
        mask = torch.tensor(grid.astype(bool), device=device)
        mean_loss = torch.nn.functional.mse_loss(
            value_tensor[mask], target_tensor[mask]
        )
        mean_loss.backward()
        optimizer.step()
