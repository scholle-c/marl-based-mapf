import pytest
import numpy as np
import torch

from marl_path.model.definition import DistanceTableCNN
from marl_path.model.feature_extraction import BasicExtractor
from marl_path.model.training import (
    compute_individual_tensors,
    compute_vdn_tensors,
    pretrain_on_bfs,
    _compute_bfs_table,
)


DEVICE = torch.device("cpu")
EXTRACTOR = BasicExtractor()

# 4x4 fully open grid
GRID = np.ones((4, 4), dtype=np.int64)

# 2 agents: start corners, goal opposite corners
STARTS = [(0, 0), (0, 3)]
GOALS = [(3, 3), (3, 0)]
SOLUTION = [
    [(0, 0), (0, 3)],
    [(1, 1), (1, 2)],
    [(2, 2), (2, 1)],
    [(3, 3), (3, 0)],
]

STARTS_1 = [(0, 0)]
GOALS_1 = [(3, 3)]
SOLUTION_1 = [[(0, 0)], [(1, 1)], [(2, 2)], [(3, 3)]]


def _model():
    return DistanceTableCNN(in_channels=EXTRACTOR.n_channels)


# ── compute_individual_tensors ────────────────────────────────────────────────


def test_individual_output_shape():
    values, targets = compute_individual_tensors(
        _model(), SOLUTION, STARTS, GOALS, GRID, DEVICE, EXTRACTOR
    )
    expected = len(STARTS) * len(SOLUTION)
    assert values.shape == (expected,)
    assert targets.shape == (expected,)


def test_individual_targets_nonnegative():
    _, targets = compute_individual_tensors(
        _model(), SOLUTION, STARTS, GOALS, GRID, DEVICE, EXTRACTOR
    )
    assert (targets >= 0).all()


def test_individual_targets_last_step_is_zero():
    """The last timestep is always at the goal, so target distance must be 0."""
    _, targets = compute_individual_tensors(
        _model(), SOLUTION, STARTS, GOALS, GRID, DEVICE, EXTRACTOR
    )
    T = len(SOLUTION)
    # Last timestep of agent 0: index T-1; last timestep of agent 1: index 2*T-1
    assert targets[T - 1].item() == 0.0
    assert targets[2 * T - 1].item() == 0.0


def test_individual_vs_vdn_lengths():
    """Individual returns num_agents*T values; VDN returns T (summed) values."""
    model = _model()
    T, N = len(SOLUTION), len(STARTS)
    v_ind, _ = compute_individual_tensors(
        model, SOLUTION, STARTS, GOALS, GRID, DEVICE, EXTRACTOR
    )
    v_vdn, _ = compute_vdn_tensors(
        model, SOLUTION, STARTS, GOALS, GRID, DEVICE, EXTRACTOR
    )
    assert v_ind.shape == (N * T,)
    assert v_vdn.shape == (T,)


def test_single_agent_individual_equals_vdn():
    """With one agent there is no decomposition, so both modes must agree."""
    model = _model()
    v_ind, t_ind = compute_individual_tensors(
        model, SOLUTION_1, STARTS_1, GOALS_1, GRID, DEVICE, EXTRACTOR
    )
    v_vdn, t_vdn = compute_vdn_tensors(
        model, SOLUTION_1, STARTS_1, GOALS_1, GRID, DEVICE, EXTRACTOR
    )
    assert torch.allclose(v_ind, v_vdn)
    assert torch.allclose(t_ind, t_vdn)


def test_individual_supports_backprop():
    model = _model()
    model.train()
    values, targets = compute_individual_tensors(
        model, SOLUTION, STARTS, GOALS, GRID, DEVICE, EXTRACTOR
    )
    torch.nn.functional.mse_loss(values, targets).backward()


# ── _compute_bfs_table ───────────────────────────────────────────────────────


def test_bfs_goal_distance_is_zero():
    goal = (2, 2)
    table = _compute_bfs_table(GRID, goal)
    assert table[goal] == 0.0


def test_bfs_distances_are_manhattan_on_open_grid():
    """On a fully open grid BFS distance equals Manhattan distance."""
    goal = (0, 0)
    table = _compute_bfs_table(GRID, goal)
    for r in range(GRID.shape[0]):
        for c in range(GRID.shape[1]):
            assert table[r, c] == r + c, f"Expected {r+c} at ({r},{c}), got {table[r,c]}"


def test_bfs_wall_cells_are_unreachable():
    grid_with_wall = np.ones((4, 4), dtype=np.int64)
    grid_with_wall[1, :] = 0  # full horizontal wall
    goal = (0, 0)
    table = _compute_bfs_table(grid_with_wall, goal)
    NIL = grid_with_wall.size
    # Cells below the wall cannot be reached
    for r in range(2, 4):
        for c in range(4):
            assert table[r, c] == NIL


# ── pretrain_on_bfs ──────────────────────────────────────────────────────────


def test_pretrain_on_bfs_runs_without_error():
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pretrain_on_bfs(model, GRID, optimizer, num_epochs=3, device=DEVICE)


def test_pretrain_on_bfs_output_is_finite():
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pretrain_on_bfs(model, GRID, optimizer, num_epochs=5, device=DEVICE)
    model.eval()
    with torch.no_grad():
        dummy_input = EXTRACTOR.extract(GRID, (0, 0), (3, 3), [], device=DEVICE)
        output = model(dummy_input)
    assert torch.isfinite(output).all()


@pytest.mark.slow
def test_pretrain_on_bfs_loss_converges():
    """Model output should approach BFS distances after sufficient training."""
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pretrain_on_bfs(model, GRID, optimizer, num_epochs=2000, device=DEVICE)

    goal = (0, 0)
    bfs_target = torch.tensor(_compute_bfs_table(GRID, goal), dtype=torch.float32)
    mask = torch.tensor(GRID.astype(bool))

    model.eval()
    with torch.no_grad():
        output = model(EXTRACTOR.extract(GRID, goal, (3, 3), [], device=DEVICE)).squeeze()
    mae = (output[mask] - bfs_target[mask]).abs().mean().item()
    assert mae < 1.5, f"Mean absolute error {mae:.2f} too high after pretraining"
