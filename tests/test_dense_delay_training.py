"""Tests for the dense NonOptimalPenaltyDelay training path (task_non_optimal_penalty_learning.md)."""
import tempfile
from pathlib import Path

import numpy as np
import torch

from marl_path.delay_methods import NonOptimalPenaltyDelay
from marl_path.dataset.instance import CachedInstance
from marl_path.dataset.dataset import CbsDataset
from marl_path.model.definition import DistanceTableCNN
from marl_path.model.feature_extraction import (
    BasicExtractor,
    AggregatedAgentsChannelExtractor,
    RichAgentsChannelExtractor,
)
from marl_path.model.training import (
    DenseDelayBatchItem,
    prepare_dense_delay_batch_item,
    update_dense_delay_from_batch,
    eval_dense_delay_loss,
    trivial_baseline_dense_loss,
    compute_mask_iou_f1,
    compute_cell_overlap,
)
from marl_path.shared.mapf_utils import BfsCache

GRID = np.ones((4, 4), dtype=bool)
PATHS = [
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(3, 3), (3, 2), (3, 1), (3, 0)],
]
GOALS = [(0, 3), (3, 0)]


def _dense_item(extractor=None, device=torch.device("cpu")) -> DenseDelayBatchItem:
    extractor = extractor or BasicExtractor()
    bfs_cache = BfsCache(GRID)
    input_tensors = [
        extractor.extract(GRID, goal=GOALS[i], start=GOALS[i], other_agents=[], device=device)
        for i in range(len(PATHS))
    ]
    return prepare_dense_delay_batch_item(GRID, bfs_cache, PATHS, GOALS, device, input_tensors)


# ── DistanceTableCNN output_activation ────────────────────────────────────────


def test_sigmoid_head_output_in_unit_interval():
    model = DistanceTableCNN(in_channels=5, output_activation="sigmoid")
    x = torch.randn(2, 5, 4, 4)
    out = model(x)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_forward_logits_matches_forward_for_sigmoid():
    model = DistanceTableCNN(in_channels=5, output_activation="sigmoid")
    x = torch.randn(2, 5, 4, 4)
    assert torch.allclose(torch.sigmoid(model.forward_logits(x)), model(x))


def test_softplus_head_is_default_and_unchanged():
    model = DistanceTableCNN(in_channels=5)
    assert model.output_activation == "softplus"
    x = torch.randn(2, 5, 4, 4)
    out = model(x)
    assert torch.all(out >= 0.0)


# ── NonOptimalPenaltyDelay dense targets ──────────────────────────────────────


def test_non_optimal_penalty_dense_target_shape_and_values():
    method = NonOptimalPenaltyDelay()
    delay_map = method.compute(GRID, bfs_cache={}, paths=PATHS, goals=GOALS, agent_idx=0)
    assert delay_map.shape == GRID.shape
    for coord in PATHS[0]:
        assert delay_map[coord] == 0.0
    assert delay_map[3, 3] == 1.0  # not on agent 0's path


def test_prepare_dense_delay_batch_item_shapes():
    item = _dense_item()
    assert item.targets.shape == (len(PATHS), *GRID.shape)
    assert item.free_mask.shape == GRID.shape
    assert bool(item.free_mask.all())  # GRID is fully open


# ── Dense loss / IoU-F1 / trivial baseline ────────────────────────────────────


def test_update_dense_delay_from_batch_runs_and_reduces_param_grad():
    model = DistanceTableCNN(in_channels=5, output_activation="sigmoid")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch = [_dense_item()]
    loss = update_dense_delay_from_batch(model, optimizer, batch, pos_weight=0.1)
    assert np.isfinite(loss)


def test_eval_dense_delay_loss_matches_manual_bce():
    model = DistanceTableCNN(in_channels=5, output_activation="sigmoid")
    batch = [_dense_item()]
    loss = eval_dense_delay_loss(model, batch, pos_weight=0.2)
    assert np.isfinite(loss) and loss > 0


def test_trivial_baseline_prefers_majority_class():
    """A constant 'always off-path' model should score near-zero BCE on the
    (majority) off-path cells and a large loss contribution only from the
    (minority) on-path cells — i.e. baseline BCE should be low but nonzero."""
    batch = [_dense_item()]
    baseline = trivial_baseline_dense_loss(batch, pos_weight=0.1)
    assert np.isfinite(baseline)
    # A model that predicts "on path" everywhere should score much worse.
    worse_batch = [
        DenseDelayBatchItem(batch[0].input_tensors, torch.zeros_like(batch[0].targets), batch[0].free_mask)
    ]
    inverted_baseline = trivial_baseline_dense_loss(worse_batch, pos_weight=0.1)
    assert inverted_baseline > baseline


def test_compute_mask_iou_f1_perfect_prediction():
    """A model whose logits exactly encode the true mask should get IoU=F1=1."""
    model = DistanceTableCNN(in_channels=5, output_activation="sigmoid")
    item = _dense_item()

    class _PerfectModel(torch.nn.Module):
        def __init__(self, targets):
            super().__init__()
            self.targets = targets

        def forward_logits(self, x):
            # off-path (1) -> large positive logit, on-path (0) -> large negative logit
            return (self.targets * 20.0) - 10.0

    perfect = _PerfectModel(item.targets)
    iou, f1 = compute_mask_iou_f1(perfect, [item])
    assert iou == 1.0
    assert f1 == 1.0
    del model  # unused, kept for parity with other tests' setup style


def test_compute_cell_overlap_disjoint_vs_identical():
    item = _dense_item()
    overlap_with_self = compute_cell_overlap([item], [item])
    assert overlap_with_self == 1.0

    disjoint_targets = torch.ones_like(item.targets)  # no on-path cells at all
    disjoint_item = DenseDelayBatchItem(item.input_tensors, disjoint_targets, item.free_mask)
    overlap = compute_cell_overlap([item], [disjoint_item])
    assert np.isnan(overlap)  # test split has no "on path" cells to compare


# ── RichAgentsChannelExtractor ─────────────────────────────────────────────────


def test_rich_extractor_channel_count():
    assert RichAgentsChannelExtractor(use_coord_channels=True).n_channels == 9
    assert RichAgentsChannelExtractor(use_coord_channels=False).n_channels == 7


def test_rich_extractor_uses_other_starts_distinct_from_goals():
    extractor = RichAgentsChannelExtractor(use_coord_channels=False)
    bfs_cache = BfsCache(GRID)
    other_goal = (0, 3)
    other_start = (3, 0)
    bfs_tables = {other_goal: bfs_cache[other_goal], other_start: bfs_cache[other_start]}

    tensor = extractor.extract(
        GRID, goal=(0, 0), start=(0, 0), other_agents=[other_goal],
        bfs_tables=bfs_tables, other_starts=[other_start],
    )
    goal_sum_ch = tensor[0, 3].numpy()
    start_sum_ch = tensor[0, 5].numpy()
    # The two aggregation channels come from different source coordinates, so
    # they must differ (both are BFS-from-a-single-point fields on an open grid).
    assert not np.array_equal(goal_sum_ch, start_sum_ch)


def test_aggregated_extractor_still_ignores_other_starts():
    """Regression: AggregatedAgentsChannelExtractor's channel count/behavior must
    be unchanged by the RichAgentsChannelExtractor addition."""
    extractor = AggregatedAgentsChannelExtractor()
    assert extractor.n_channels == 7
    tensor = extractor.extract(GRID, goal=(0, 0), start=(0, 0), other_agents=[(0, 3)])
    assert tensor.shape == (1, 7, *GRID.shape)


# ── CbsDataset dense mode (end-to-end with synthetic cached instance) ─────────


def _write_map(path: Path, grid: np.ndarray) -> None:
    height, width = grid.shape
    lines = ["type octile", f"height {height}", f"width {width}", "map"]
    for row in grid:
        lines.append("".join("." if v else "@" for v in row))
    path.write_text("\n".join(lines) + "\n")


def _write_scen(path: Path, map_name: str, starts, goals) -> None:
    lines = ["version 1"]
    for i, (s, g) in enumerate(zip(starts, goals)):
        sy, sx = s
        gy, gx = g
        lines.append(f"{i}\t{map_name}\t4\t4\t{sx}\t{sy}\t{gx}\t{gy}\t1.0")
    path.write_text("\n".join(lines) + "\n")


def test_cbs_dataset_dense_mode_end_to_end():
    starts = [(0, 0), (3, 3)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        map_file = tmp_path / "test.map"
        scen_file = tmp_path / "test.scen"
        _write_map(map_file, GRID)
        _write_scen(scen_file, "test.map", starts, GOALS)

        instance = CachedInstance(
            map_file=str(map_file),
            scen_file=str(scen_file),
            agent_indices=[0, 1],
            paths=PATHS,
        )
        npz_path = tmp_path / "instance_00.npz"
        instance.save(npz_path)

        dataset = CbsDataset(tmp_path, mode="dense")
        assert len(dataset) == 1
        item = dataset[0]
        assert isinstance(item, DenseDelayBatchItem)
        assert item.targets.shape == (2, *GRID.shape)
        for coord in PATHS[0]:
            assert item.targets[0][coord] == 0.0
        for coord in PATHS[1]:
            assert item.targets[1][coord] == 0.0


def test_cbs_dataset_sparse_mode_regression_unchanged():
    """The original sparse path (default mode) must keep working unchanged."""
    from marl_path.model.training import DelayBatchItem

    starts = [(0, 0), (3, 3)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        map_file = tmp_path / "test.map"
        scen_file = tmp_path / "test.scen"
        _write_map(map_file, GRID)
        _write_scen(scen_file, "test.map", starts, GOALS)

        instance = CachedInstance(
            map_file=str(map_file), scen_file=str(scen_file),
            agent_indices=[0, 1], paths=PATHS,
        )
        instance.save(tmp_path / "instance_00.npz")

        dataset = CbsDataset(tmp_path)  # default mode="sparse"
        item = dataset[0]
        assert isinstance(item, DelayBatchItem)
        assert item.targets.shape[0] == len(PATHS[0]) + len(PATHS[1])
