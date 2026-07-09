"""Model training utilities."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Sequence
import torch
import numpy as np

from marl_path.shared import Coord, get_neighbors
from marl_path.shared.mapf_utils import BfsCache
from marl_path.delay_methods import DelayMethod, NonOptimalPenaltyDelay
from .feature_extraction import FeatureExtractor, BasicExtractor


@dataclass
class DelayBatchItem:
    """Raw data for one episode needed to compute loss with gradients."""

    input_tensors: List[torch.Tensor]
    paths: List[List[Coord]]
    bfs_distances: List[np.ndarray]
    targets: torch.Tensor


@dataclass
class DenseDelayBatchItem:
    """Raw data for one episode's dense, full-grid target (e.g. NonOptimalPenaltyDelay).

    Unlike DelayBatchItem, targets cover every free cell of the grid for every
    agent, not just cells visited on that agent's path.
    """

    input_tensors: List[torch.Tensor]
    targets: torch.Tensor  # (num_agents, H, W)
    free_mask: torch.Tensor  # (H, W) bool — True where the grid is traversable


def prepare_dense_delay_batch_item(
    grid: np.ndarray,
    bfs_cache: BfsCache,
    paths: List[List[Coord]],
    goals: List[Coord],
    device: torch.device,
    input_tensors: List[torch.Tensor],
    delay_method: DelayMethod | None = None,
) -> DenseDelayBatchItem:
    """Build full-grid, per-agent targets for one episode (all agents).

    delay_method defaults to NonOptimalPenaltyDelay: 0 on the agent's CBS-optimal
    path, 1 everywhere else. Any DelayMethod that returns a full grid.shape array
    works here.
    """
    method = delay_method or NonOptimalPenaltyDelay()
    dense = np.stack(
        [
            method.compute(grid, bfs_cache, paths, goals, agent_idx=i)
            for i in range(len(paths))
        ]
    ).astype(np.float32)
    targets = torch.from_numpy(dense).to(device)
    free_mask = torch.from_numpy(grid.astype(bool)).to(device)
    return DenseDelayBatchItem(input_tensors, targets, free_mask)


def prepare_delay_batch_item(
    _model: Any,
    solution: Any,
    starts: Any,
    device: torch.device,
    bfs_tables: List[np.ndarray],
    input_tensors: List[torch.Tensor],
) -> DelayBatchItem:
    """Collect paths, BFS distances, and targets for one episode without running
    a forward pass. The forward pass is deferred to update_delay_from_batch so
    that computation graphs are not held across the entire batch accumulation."""
    num_agents = len(starts)
    paths = []
    bfs_distances = []
    all_targets: list[float] = []

    for agent_idx in range(num_agents):
        path = [solution[t][agent_idx] for t in range(len(solution))]
        paths.append(path)
        bfs_distances.append(_get_via_coordinates(bfs_tables[agent_idx], path))
        all_targets.extend(_get_path_target_first_visit(path))

    target_tensor = torch.tensor(all_targets, dtype=torch.float32, device=device)
    return DelayBatchItem(input_tensors, paths, bfs_distances, target_tensor)


def update_delay_from_batch(
    model: Any,
    optimizer: Any,
    batch: Sequence[DelayBatchItem],
    loss_fn=torch.nn.functional.mse_loss,
) -> float:
    """Recompute forward passes with gradients for each item in the batch,
    accumulate gradients, then step the optimizer once."""
    if not batch:
        return float("nan")
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for item in batch:
        delay_tables = _batch_delay_tables(model, item.input_tensors)

        all_values: list[torch.Tensor] = []
        for agent_idx, (path, bfs_dist) in enumerate(
            zip(item.paths, item.bfs_distances)
        ):
            predicted_delay = _get_via_coordinates(delay_tables[agent_idx], path)
            bfs_tensor = torch.tensor(
                bfs_dist, dtype=torch.float32, device=delay_tables.device
            )
            all_values.append(predicted_delay + bfs_tensor)

        values = torch.cat(all_values)
        loss = loss_fn(values, item.targets)
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss / len(batch)


def eval_delay_loss(
    model: Any,
    batch: Sequence[DelayBatchItem],
    loss_fn=torch.nn.functional.mse_loss,
) -> float:
    """Compute mean loss over a batch without updating model weights."""
    if not batch:
        return float("nan")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for item in batch:
            delay_tables = _batch_delay_tables(model, item.input_tensors)
            all_values: list[torch.Tensor] = []
            for agent_idx, (path, bfs_dist) in enumerate(
                zip(item.paths, item.bfs_distances)
            ):
                predicted_delay = _get_via_coordinates(delay_tables[agent_idx], path)
                bfs_tensor = torch.tensor(
                    bfs_dist, dtype=torch.float32, device=delay_tables.device
                )
                all_values.append(predicted_delay + bfs_tensor)
            values = torch.cat(all_values)
            total_loss += loss_fn(values, item.targets).item()
    return total_loss / len(batch)


def _batch_delay_tables(model: Any, input_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Single batched forward pass for all agents."""
    batched = torch.cat(input_tensors, dim=0)  # (num_agents, C, H, W)
    return model(batched).squeeze(1)  # (num_agents, H, W)


def _batch_delay_logits(model: Any, input_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Single batched forward pass returning pre-activation logits (for BCEWithLogitsLoss)."""
    batched = torch.cat(input_tensors, dim=0)  # (num_agents, C, H, W)
    return model.forward_logits(batched).squeeze(1)  # (num_agents, H, W)


def update_dense_delay_from_batch(
    model: Any,
    optimizer: Any,
    batch: Sequence[DenseDelayBatchItem],
    pos_weight: float = 0.05,
) -> float:
    """Dense counterpart of update_delay_from_batch: BCEWithLogitsLoss over every
    free cell of the grid (not just path cells), for every agent.

    pos_weight scales the majority class (label 1 = "off optimal path") down
    relative to the minority class (label 0 = "on path"), since that majority
    otherwise dominates the loss. Start with inverse class frequency and tune.
    """
    if not batch:
        return float("nan")
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for item in batch:
        logits = _batch_delay_logits(model, item.input_tensors)
        mask = item.free_mask.unsqueeze(0).expand_as(logits)
        weight = torch.tensor(pos_weight, device=logits.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask], item.targets[mask], pos_weight=weight
        )
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss / len(batch)


def eval_dense_delay_loss(
    model: Any,
    batch: Sequence[DenseDelayBatchItem],
    pos_weight: float = 0.05,
) -> float:
    """Compute mean dense BCE loss over a batch without updating model weights."""
    if not batch:
        return float("nan")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for item in batch:
            logits = _batch_delay_logits(model, item.input_tensors)
            mask = item.free_mask.unsqueeze(0).expand_as(logits)
            weight = torch.tensor(pos_weight, device=logits.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[mask], item.targets[mask], pos_weight=weight
            )
            total_loss += loss.item()
    return total_loss / len(batch)


def trivial_baseline_dense_loss(
    batch: Sequence[DenseDelayBatchItem],
    pos_weight: float = 0.05,
) -> float:
    """BCE loss of a constant "always predict off-path" model — the class-imbalance
    floor that any trained model's loss should be compared against."""
    if not batch:
        return float("nan")
    total_loss = 0.0
    for item in batch:
        mask = item.free_mask.unsqueeze(0).expand_as(item.targets)
        logits = torch.full_like(item.targets, 10.0)  # sigmoid(10) ~= 1.0
        weight = torch.tensor(pos_weight, device=item.targets.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask], item.targets[mask], pos_weight=weight
        )
        total_loss += loss.item()
    return total_loss / len(batch)


def compute_mask_iou_f1(
    model: Any,
    batch: Sequence[DenseDelayBatchItem],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """IoU/F1 between the predicted "on-path" mask (sigmoid output < threshold) and
    the true CBS-path mask (target == 0), over free cells only, averaged over every
    agent/instance in the batch.

    This is the class-imbalance-robust metric: a model that always predicts
    "off-path" gets BCE that looks deceptively good but IoU/F1 == 0 here.
    """
    if not batch:
        return float("nan"), float("nan")
    model.eval()
    ious: list[float] = []
    f1s: list[float] = []
    with torch.no_grad():
        for item in batch:
            logits = _batch_delay_logits(model, item.input_tensors)
            pred_on_path = torch.sigmoid(logits) < threshold
            true_on_path = item.targets < 0.5
            mask = item.free_mask.unsqueeze(0).expand_as(pred_on_path)
            for agent_idx in range(pred_on_path.shape[0]):
                p = pred_on_path[agent_idx][mask[agent_idx]]
                t = true_on_path[agent_idx][mask[agent_idx]]
                tp = int((p & t).sum())
                fp = int((p & ~t).sum())
                fn = int((~p & t).sum())
                iou_denom = tp + fp + fn
                ious.append(tp / iou_denom if iou_denom else 1.0)
                f1_denom = 2 * tp + fp + fn
                f1s.append(2 * tp / f1_denom if f1_denom else 1.0)
    return float(np.mean(ious)), float(np.mean(f1s))


def compute_cell_overlap(
    train_batch: Sequence[DenseDelayBatchItem],
    test_batch: Sequence[DenseDelayBatchItem],
) -> float:
    """Fraction of free cells visited (target == 0, "on path") in the test split
    that were also visited somewhere in the train split.

    High overlap plus high test IoU/F1 is a warning sign of memorization rather
    than generalization (see task's train/test diagnostic requirement).
    """

    def _visited_cells(batch: Sequence[DenseDelayBatchItem]) -> set[tuple[int, int]]:
        visited: set[tuple[int, int]] = set()
        for item in batch:
            on_path = (item.targets < 0.5).cpu().numpy()
            for agent_mask in on_path:
                ys, xs = np.nonzero(agent_mask)
                visited.update(zip(ys.tolist(), xs.tolist()))
        return visited

    train_cells = _visited_cells(train_batch)
    test_cells = _visited_cells(test_batch)
    if not test_cells:
        return float("nan")
    return len(test_cells & train_cells) / len(test_cells)


def _get_path_target_first_visit(path: Any) -> List[int]:
    """Computes first-visit timestep targets for each cell on the path.
    target(v) = total_path_length - t_first_visit(v)
    """
    total_length = len(path) - 1

    first_visit: dict = {}
    for t, coord in enumerate(path):
        if coord not in first_visit:
            first_visit[coord] = t

    targets = []
    for coord in path:
        targets.append(total_length - first_visit[coord])

    return targets


def _get_via_coordinates(arr: Any, coords: List[Coord]) -> Any:
    """Access elements of a 2D array via a list of (y,x) coordinates."""
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

        mask = torch.tensor(grid.astype(bool), device=device)
        mean_loss = torch.nn.functional.mse_loss(
            value_tensor[mask], target_tensor[mask]
        )
        mean_loss.backward()
        optimizer.step()
