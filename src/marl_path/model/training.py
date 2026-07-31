"""Model training utilities."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence
import torch
import numpy as np

from marl_path.shared import Coord, get_neighbors
from marl_path.shared.mapf_utils import BfsCache
from marl_path.delay_methods import DelayMethod, NonOptimalPenaltyDelay
from .feature_extraction import FeatureExtractor, BasicExtractor


@dataclass
class DenseDelayBatchItem:
    """Raw data for one episode's dense, full-grid target (e.g. NonOptimalPenaltyDelay).

    Targets cover every free cell of the grid for every agent, not just cells
    visited on that agent's path.
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
    """BCEWithLogitsLoss over every free cell of the grid (not just path cells),
    for every agent.

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
    batch: Iterable[DenseDelayBatchItem],
    pos_weight: float = 0.05,
) -> float:
    """Compute mean dense BCE loss over a batch without updating model weights.

    `batch` is consumed lazily (a generator over a dataset works fine) so a full
    split never needs to be materialized in memory at once.
    """
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for item in batch:
            logits = _batch_delay_logits(model, item.input_tensors)
            mask = item.free_mask.unsqueeze(0).expand_as(logits)
            weight = torch.tensor(pos_weight, device=logits.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[mask], item.targets[mask], pos_weight=weight
            )
            total_loss += loss.item()
            n += 1
    return total_loss / n if n else float("nan")


# ── Regression (continuous target, e.g. dense CBS funnel) ─────────────────────


def _batch_delay_values(model: Any, input_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Single batched forward pass returning activated per-cell values.

    Uses model.forward (softplus head for regression models), so training and
    LaCAM inference apply the exact same output transform.
    """
    batched = torch.cat(input_tensors, dim=0)  # (num_agents, C, H, W)
    return model(batched).squeeze(1)  # (num_agents, H, W)


def _regression_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_type: str
) -> torch.Tensor:
    if loss_type == "mse":
        return torch.nn.functional.mse_loss(pred, target)
    if loss_type == "huber":
        return torch.nn.functional.smooth_l1_loss(pred, target)
    raise ValueError(f"Unknown loss_type {loss_type!r}; choose 'mse' or 'huber'")


def update_dense_delay_regression_from_batch(
    model: Any,
    optimizer: Any,
    batch: Sequence[DenseDelayBatchItem],
    loss_type: str = "mse",
) -> float:
    """Regression loss over every free cell of the grid, for every agent.

    Counterpart of update_dense_delay_from_batch for continuous targets: no
    sigmoid, no class weighting — just MSE/Huber between the softplus output
    and the dense target (e.g. hop-distance CBS funnel)."""
    if not batch:
        return float("nan")
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for item in batch:
        preds = _batch_delay_values(model, item.input_tensors)
        mask = item.free_mask.unsqueeze(0).expand_as(preds)
        loss = _regression_loss(preds[mask], item.targets[mask], loss_type)
        loss.backward()
        total_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss / len(batch)


def eval_dense_delay_regression_loss(
    model: Any,
    batch: Iterable[DenseDelayBatchItem],
    loss_type: str = "mse",
) -> float:
    """Mean regression loss over free cells without updating weights."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for item in batch:
            preds = _batch_delay_values(model, item.input_tensors)
            mask = item.free_mask.unsqueeze(0).expand_as(preds)
            total_loss += _regression_loss(
                preds[mask], item.targets[mask], loss_type
            ).item()
            n += 1
    return total_loss / n if n else float("nan")


def compute_regression_metrics(
    model: Any, batch: Iterable[DenseDelayBatchItem]
) -> tuple[float, float]:
    """(MAE, RMSE) over free cells, averaged per item — interpretable in the
    target's own units (hops), unlike the raw MSE."""
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0
    with torch.no_grad():
        for item in batch:
            preds = _batch_delay_values(model, item.input_tensors)
            mask = item.free_mask.unsqueeze(0).expand_as(preds)
            diff = preds[mask] - item.targets[mask]
            mae_sum += diff.abs().mean().item()
            mse_sum += (diff**2).mean().item()
            n += 1
    if not n:
        return float("nan"), float("nan")
    return mae_sum / n, (mse_sum / n) ** 0.5


def trivial_baseline_regression_loss(
    batch: Iterable[DenseDelayBatchItem], loss_type: str = "mse"
) -> float:
    """Loss of the constant per-item mean predictor — the variance floor that a
    trained regressor must beat to have learned any spatial structure."""
    total_loss = 0.0
    n = 0
    for item in batch:
        mask = item.free_mask.unsqueeze(0).expand_as(item.targets)
        target = item.targets[mask]
        pred = torch.full_like(target, float(target.mean()))
        total_loss += _regression_loss(pred, target, loss_type).item()
        n += 1
    return total_loss / n if n else float("nan")


def trivial_baseline_dense_loss(
    batch: Iterable[DenseDelayBatchItem],
    pos_weight: float = 0.05,
) -> float:
    """BCE loss of a constant "always predict off-path" model — the class-imbalance
    floor that any trained model's loss should be compared against."""
    total_loss = 0.0
    n = 0
    for item in batch:
        mask = item.free_mask.unsqueeze(0).expand_as(item.targets)
        logits = torch.full_like(item.targets, 10.0)  # sigmoid(10) ~= 1.0
        weight = torch.tensor(pos_weight, device=item.targets.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask], item.targets[mask], pos_weight=weight
        )
        total_loss += loss.item()
        n += 1
    return total_loss / n if n else float("nan")


def compute_mask_iou_f1(
    model: Any,
    batch: Iterable[DenseDelayBatchItem],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """IoU/F1 between the predicted "on-path" mask (sigmoid output < threshold) and
    the true CBS-path mask (target == 0), over free cells only, averaged over every
    agent/instance in the batch.

    This is the class-imbalance-robust metric: a model that always predicts
    "off-path" gets BCE that looks deceptively good but IoU/F1 == 0 here.
    """
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
    if not ious:
        return float("nan"), float("nan")
    return float(np.mean(ious)), float(np.mean(f1s))


def compute_cell_overlap(
    train_batch: Iterable[DenseDelayBatchItem],
    test_batch: Iterable[DenseDelayBatchItem],
) -> float:
    """Fraction of free cells visited (target == 0, "on path") in the test split
    that were also visited somewhere in the train split.

    High overlap plus high test IoU/F1 is a warning sign of memorization rather
    than generalization (see task's train/test diagnostic requirement).
    """

    def _visited_cells(batch: Iterable[DenseDelayBatchItem]) -> set[tuple[int, int]]:
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
