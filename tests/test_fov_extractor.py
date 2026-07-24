"""Tests for the FOV-restricted extractor/model/training path (FovPathExtractor,
FovPatchTransformer, sinusoidal_encoding_2d, Fov*DelayBatchItem training/eval)."""
import tempfile
from pathlib import Path

import numpy as np
import torch

from marl_path.model.definition import FovPatchTransformer, sinusoidal_encoding_2d
from marl_path.model.feature_extraction import FovPathExtractor
from marl_path.model.training import (
    FovDelayBatchItem,
    prepare_fov_delay_batch_item,
    update_fov_delay_from_batch,
    eval_fov_delay_loss,
    trivial_baseline_fov_loss,
    compute_fov_mask_iou_f1,
    compute_fov_cell_overlap,
)
from marl_path.delay_methods import NonOptimalPenaltyDelay
from marl_path.dataset.instance import CachedInstance
from marl_path.dataset.dataset import CbsDataset
from marl_path.shared.mapf_utils import BfsCache
from marl_path.pycam.dist_table import DistTable

GRID = np.ones((10, 10), dtype=bool)
OWN_START = (0, 0)
GOAL = (0, 9)


# ── FovPathExtractor: FOV cell selection ──────────────────────────────────────


def test_n_channels():
    assert FovPathExtractor(include_intersection=False).n_channels == 6
    assert FovPathExtractor(include_intersection=True).n_channels == 7


def test_fov_excludes_cells_far_from_own_path():
    bfs_cache = BfsCache(GRID)
    extractor = FovPathExtractor(fov_radius=1)
    tokens = extractor.extract_tokens(
        GRID, goal=GOAL, other_agents=[], other_starts=[], own_start=OWN_START,
        bfs_tables={GOAL: bfs_cache[GOAL]},
    )
    coords = {tuple(c) for c in tokens.coords.tolist()}
    # own_start/goal must always be in-FOV (they're on the agent's own path).
    assert OWN_START in coords
    assert GOAL in coords
    # A cell far below the (row-0) path is outside a radius-1 band.
    assert (5, 5) not in coords
    assert not tokens.has_other_agents


def test_fov_includes_intersecting_other_agent_and_excludes_distant_one():
    bfs_cache = BfsCache(GRID)
    extractor = FovPathExtractor(fov_radius=1)
    # Agent B crosses the row-0/1 band near x=5 -> should be pulled in.
    near_goal, near_start = (1, 5), (2, 5)
    # Agent C stays far away (row 8) -> should be dropped entirely, not zeroed.
    far_goal, far_start = (8, 0), (8, 9)
    bfs_tables = {
        GOAL: bfs_cache[GOAL],
        near_goal: bfs_cache[near_goal],
        far_goal: bfs_cache[far_goal],
    }
    tokens = extractor.extract_tokens(
        GRID, goal=GOAL, other_agents=[near_goal, far_goal],
        other_starts=[near_start, far_start], own_start=OWN_START,
        bfs_tables=bfs_tables,
    )
    assert tokens.has_other_agents
    coords = {tuple(c) for c in tokens.coords.tolist()}
    other_mask_col = tokens.features[:, 4].numpy()  # other_paths_mask channel
    other_mask_present = {
        c for c, m in zip(coords, other_mask_col) if m > 0
    }
    assert len(other_mask_present) > 0
    # None of agent C's (far, row-8) cells should ever appear in the FOV at all.
    assert not any(c[0] >= 7 for c in coords)


def test_fov_empty_without_own_start_or_bfs_tables():
    extractor = FovPathExtractor()
    tokens = extractor.extract_tokens(
        GRID, goal=GOAL, other_agents=[], other_starts=[], own_start=None,
        bfs_tables=None,
    )
    assert tokens.coords.shape == (0, 2)
    assert tokens.features.shape == (0, extractor.n_channels)


# ── sinusoidal_encoding_2d ─────────────────────────────────────────────────────


def test_sinusoidal_encoding_shape_and_determinism():
    coords = torch.tensor([[0, 0], [3, 7], [9, 9]], dtype=torch.long)
    pe1 = sinusoidal_encoding_2d(coords, embed_dim=16)
    pe2 = sinusoidal_encoding_2d(coords, embed_dim=16)
    assert pe1.shape == (3, 16)
    assert torch.equal(pe1, pe2)  # stateless: identical inputs -> identical output
    assert not torch.equal(pe1[0], pe1[1])  # distinct coords -> distinct encodings


def test_sinusoidal_encoding_has_no_parameters():
    # Confirms this really is a stateless function, not a disguised nn.Module —
    # the whole point of replacing PatchTransformer's learned nn.Parameter table.
    import inspect
    assert not inspect.isclass(sinusoidal_encoding_2d)
    coords = torch.zeros((1, 2), dtype=torch.long)
    out = sinusoidal_encoding_2d(coords, embed_dim=8)
    assert out.requires_grad is False


def test_sinusoidal_encoding_requires_embed_dim_divisible_by_4():
    coords = torch.zeros((1, 2), dtype=torch.long)
    try:
        sinusoidal_encoding_2d(coords, embed_dim=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ── FovPatchTransformer ─────────────────────────────────────────────────────


def test_fov_transformer_output_shape_and_range():
    model = FovPatchTransformer(in_channels=6, embed_dim=8, num_layers=1, num_heads=2)
    features = torch.randn(2, 5, 6)
    coords = torch.randint(0, 10, (2, 5, 2))
    out = model(features, coords)
    assert out.shape == (2, 5)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_fov_transformer_forward_logits_matches_forward():
    model = FovPatchTransformer(in_channels=6, embed_dim=8, num_layers=1, num_heads=2)
    features = torch.randn(1, 3, 6)
    coords = torch.randint(0, 10, (1, 3, 2))
    assert torch.allclose(
        torch.sigmoid(model.forward_logits(features, coords)), model(features, coords)
    )


def test_fov_transformer_padding_mask_isolates_real_tokens():
    """Changing a padded token's content must not change the prediction for the
    real (unpadded) tokens in the same sequence — otherwise padding would leak
    into the attention computation."""
    model = FovPatchTransformer(in_channels=6, embed_dim=8, num_layers=2, num_heads=2)
    model.eval()
    real_features = torch.randn(1, 3, 6)
    real_coords = torch.randint(0, 10, (1, 3, 2))
    pad_features_a = torch.cat([real_features, torch.zeros(1, 2, 6)], dim=1)
    pad_features_b = torch.cat([real_features, torch.randn(1, 2, 6) * 99], dim=1)
    coords_padded = torch.cat([real_coords, torch.zeros(1, 2, 2, dtype=torch.long)], dim=1)
    mask = torch.tensor([[False, False, False, True, True]])

    with torch.no_grad():
        out_a = model.forward_logits(pad_features_a, coords_padded, padding_mask=mask)
        out_b = model.forward_logits(pad_features_b, coords_padded, padding_mask=mask)

    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-5)


# ── prepare_fov_delay_batch_item / loss / eval ────────────────────────────────


def _fov_batch_item() -> FovDelayBatchItem:
    grid = np.ones((6, 6), dtype=bool)
    paths = [[(0, 0), (0, 1), (0, 2)], [(5, 5), (5, 4), (5, 3)]]
    goals = [(0, 2), (5, 3)]
    bfs_cache = BfsCache(grid)
    extractor = FovPathExtractor(fov_radius=1)
    tokens_per_agent = [
        extractor.extract_tokens(
            grid, goal=goals[i], other_agents=[], other_starts=[],
            own_start=paths[i][0], bfs_tables={goals[i]: bfs_cache[goals[i]]},
        )
        for i in range(2)
    ]
    return prepare_fov_delay_batch_item(
        grid, bfs_cache, paths, goals, torch.device("cpu"), tokens_per_agent,
        NonOptimalPenaltyDelay(),
    )


def test_prepare_fov_delay_batch_item_shapes_and_targets():
    item = _fov_batch_item()
    assert item.features.shape[0] == 2  # num_agents
    assert item.coords.shape == item.features.shape[:2] + (2,)
    assert item.targets.shape == item.padding_mask.shape == item.features.shape[:2]
    # Own-path cells must have target 0 (on optimal path) wherever unpadded.
    valid = ~item.padding_mask
    on_path_targets = item.targets[0][valid[0]]
    assert (on_path_targets == 0.0).any()  # at least the agent's own path cells


def test_update_and_eval_fov_delay_loss_runs():
    model = FovPatchTransformer(in_channels=6, embed_dim=8, num_layers=1, num_heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch = [_fov_batch_item()]
    loss = update_fov_delay_from_batch(model, optimizer, batch, pos_weight=0.1)
    assert np.isfinite(loss)
    eval_loss = eval_fov_delay_loss(model, batch, pos_weight=0.1)
    assert np.isfinite(eval_loss) and eval_loss > 0


def test_trivial_baseline_fov_loss_finite():
    batch = [_fov_batch_item()]
    baseline = trivial_baseline_fov_loss(batch, pos_weight=0.1)
    assert np.isfinite(baseline)


def test_compute_fov_mask_iou_f1_perfect_prediction():
    item = _fov_batch_item()

    class _PerfectModel(torch.nn.Module):
        def __init__(self, targets):
            super().__init__()
            self.targets = targets

        def forward_logits(self, features, coords, padding_mask=None):
            return (self.targets * 20.0) - 10.0

    perfect = _PerfectModel(item.targets)
    iou, f1 = compute_fov_mask_iou_f1(perfect, [item])
    assert iou == 1.0
    assert f1 == 1.0


def test_compute_fov_cell_overlap_identical_is_one():
    item = _fov_batch_item()
    assert compute_fov_cell_overlap([item], [item]) == 1.0


# ── DistTable integration: FOV scatter-back ────────────────────────────────────


def test_dist_table_fov_delay_zero_outside_fov_and_matches_model_inside():
    grid = GRID
    bfs_cache = BfsCache(grid)
    extractor = FovPathExtractor(fov_radius=1)
    model = FovPatchTransformer(in_channels=6, embed_dim=8, num_layers=1, num_heads=2)
    model.eval()

    dt = DistTable(
        grid, GOAL, model=model, device=torch.device("cpu"), extractor=extractor,
        other_agents=[], other_agent_starts=[], own_start=OWN_START,
        bfs_cache=bfs_cache,
    )

    # A cell far from the row-0 path must be untouched (delay=0, pure BFS).
    assert dt.delay[(5, 5)] == 0.0

    tokens = extractor.extract_tokens(
        grid, goal=GOAL, other_agents=[], other_starts=[], own_start=OWN_START,
        bfs_tables={GOAL: bfs_cache[GOAL]},
    )
    with torch.no_grad():
        expected = model(tokens.features.unsqueeze(0), tokens.coords.unsqueeze(0)).squeeze(0).numpy()
    for (y, x), val in zip(tokens.coords.tolist(), expected):
        assert abs(dt.delay[y, x] - val) < 1e-5


# ── CbsDataset FOV mode end-to-end ─────────────────────────────────────────────


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


def test_cbs_dataset_fov_mode_end_to_end():
    grid = np.ones((4, 4), dtype=bool)
    paths = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(3, 3), (3, 2), (3, 1), (3, 0)],
    ]
    goals = [(0, 3), (3, 0)]
    starts = [(0, 0), (3, 3)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        map_file = tmp_path / "test.map"
        scen_file = tmp_path / "test.scen"
        _write_map(map_file, grid)
        _write_scen(scen_file, "test.map", starts, goals)

        instance = CachedInstance(
            map_file=str(map_file), scen_file=str(scen_file),
            agent_indices=[0, 1], paths=paths,
        )
        npz_path = tmp_path / "instance_00.npz"
        instance.save(npz_path)

        dataset = CbsDataset(tmp_path, extractor=FovPathExtractor(fov_radius=1))
        assert len(dataset) == 1
        item = dataset[0]
        assert isinstance(item, FovDelayBatchItem)
        assert item.features.shape[0] == 2
        assert item.features.shape[2] == FovPathExtractor(fov_radius=1).n_channels
