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


def compute_delay_tensors(
    model: Any,
    solution: Any,
    starts: Any,
    device: torch.device,
    bfs_tables: List[np.ndarray],
    input_tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like compute_vdn_tensors but without VDN decomposition: each agent's
    predicted values and targets are concatenated rather than summed, so the
    loss is computed independently per agent per timestep."""
    num_agents = len(starts)
    delay_tables = _batch_delay_tables(model, input_tensors)

    all_values: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for agent_idx in range(num_agents):
        path = [solution[t][agent_idx] for t in range(len(solution))]
        predicted_delay = _get_via_coordinates(delay_tables[agent_idx], path)
        bfs_distance = _get_via_coordinates(bfs_tables[agent_idx], path)
        bfs_tensor = torch.tensor(bfs_distance, dtype=torch.float32, device=device)
        values = predicted_delay + bfs_tensor
        all_values.append(values)
        all_targets.append(
            torch.tensor(
                _get_path_target_first_visit(path), dtype=torch.float32, device=device
            )
        )

    return torch.cat(all_values), torch.cat(all_targets)


def _batch_delay_tables(model: Any, input_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Single batched forward pass for all agents."""
    batched = torch.cat(input_tensors, dim=0)  # (num_agents, C, H, W)
    return model(batched).squeeze(1)  # (num_agents, H, W)


def update_from_batch(
    model: Any,
    optimizer: Any,
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    loss_fn=torch.nn.functional.mse_loss,
    weights: list[float] | None = None,
) -> float:
    if not batch:
        return float("nan")
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    i = 0
    for values, targets in batch:
        loss = loss_fn(values, targets)
        if weights is not None:
            loss = loss * weights[i]
        loss.backward()
        total_loss += loss.item()
        i += 1
    # If magnitude of gradients exceeds 1.0, gradients are scaled down to magnitude = 1.0. Should prevent bad updates pushing the model in a bad direction.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss / len(batch)


def _get_path_target_first_visit(path: Any) -> List[int]:
    """
    Computes first-visit timestep targets for each cell on the path.
    target(v) = total_path_length - t_first_visit(v)
    """
    total_length = len(path) - 1  # remaining steps from start

    first_visit: dict = {}
    for t, coord in enumerate(path):
        if coord not in first_visit:
            first_visit[coord] = t

    targets = []
    for coord in path:
        targets.append(total_length - first_visit[coord])

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
