"""Tests for the fn_penalty asymmetric-loss weighting (dense and FOV paths).

Motivated by the label-noise sweep: false negatives on true on-path cells
(target == 0) collapse LaCAM's downstream search far more than false
positives on neighboring off-path cells, at matched error rates. fn_penalty
up-weights BCE errors on target == 0 cells specifically, on top of
pos_weight's class-balance correction.
"""
import numpy as np
import torch

from marl_path.model.definition import DistanceTableCNN
from marl_path.model.feature_extraction import BasicExtractor
from marl_path.model.training import (
    DenseDelayBatchItem,
    prepare_dense_delay_batch_item,
    update_dense_delay_from_batch,
    eval_dense_delay_loss,
    trivial_baseline_dense_loss,
    _fn_penalty_weight,
)
from marl_path.shared.mapf_utils import BfsCache

GRID = np.ones((4, 4), dtype=bool)
PATHS = [
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(3, 3), (3, 2), (3, 1), (3, 0)],
]
GOALS = [(0, 3), (3, 0)]


def _dense_item() -> DenseDelayBatchItem:
    extractor = BasicExtractor(use_coord_channels=False)
    bfs_cache = BfsCache(GRID)
    input_tensors = [
        extractor.extract(GRID, goal=GOALS[i], start=GOALS[i], other_agents=[])
        for i in range(len(PATHS))
    ]
    return prepare_dense_delay_batch_item(
        GRID, bfs_cache, PATHS, GOALS, torch.device("cpu"), input_tensors
    )


# ── _fn_penalty_weight ────────────────────────────────────────────────────────


def test_fn_penalty_weight_none_when_default():
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert _fn_penalty_weight(targets, fn_penalty=1.0) is None


def test_fn_penalty_weight_upweights_only_on_path_cells():
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    weight = _fn_penalty_weight(targets, fn_penalty=3.0)
    assert weight is not None
    assert torch.equal(weight, torch.tensor([3.0, 1.0, 3.0, 1.0]))


# ── Dense loss functions: fn_penalty wiring ───────────────────────────────────


def test_fn_penalty_default_matches_unweighted_loss():
    """fn_penalty=1.0 must reproduce the exact previous (pos_weight-only) loss —
    no behavior change for existing callers that don't pass fn_penalty."""
    item = _dense_item()
    model = DistanceTableCNN(in_channels=3)
    loss_default = eval_dense_delay_loss(model, [item], pos_weight=0.1)
    loss_explicit = eval_dense_delay_loss(model, [item], pos_weight=0.1, fn_penalty=1.0)
    assert loss_default == loss_explicit


def test_fn_penalty_increases_loss_for_on_path_errors():
    """A model that is wrong specifically on on-path cells should get a larger
    loss increase under fn_penalty>1 than a model wrong on off-path cells."""
    item = _dense_item()

    class _WrongOnPath(torch.nn.Module):
        def forward_logits(self, x):
            # Predicts everything as strongly "off-path" -> wrong exactly on
            # the on-path (target=0) cells, right on off-path (target=1) cells.
            return torch.full((x.shape[0], 4, 4), 10.0)

    class _WrongOffPath(torch.nn.Module):
        def forward_logits(self, x):
            # Predicts everything as strongly "on-path" -> wrong exactly on
            # the off-path (target=1) cells, right on on-path (target=0) cells.
            return torch.full((x.shape[0], 4, 4), -10.0)

    wrong_on_path = _WrongOnPath()
    wrong_off_path = _WrongOffPath()

    base_on = eval_dense_delay_loss(wrong_on_path, [item], pos_weight=1.0, fn_penalty=1.0)
    boosted_on = eval_dense_delay_loss(wrong_on_path, [item], pos_weight=1.0, fn_penalty=5.0)
    base_off = eval_dense_delay_loss(wrong_off_path, [item], pos_weight=1.0, fn_penalty=1.0)
    boosted_off = eval_dense_delay_loss(wrong_off_path, [item], pos_weight=1.0, fn_penalty=5.0)

    # fn_penalty=5 must scale up the on-path-error loss by ~5x (all errors are
    # on target==0 cells for this model)...
    assert boosted_on > base_on
    assert abs(boosted_on / base_on - 5.0) < 1e-3
    # ...but leave the off-path-error loss practically unchanged (all real
    # errors are on target==1 cells, which fn_penalty never touches — the
    # tiny residual here is just float noise from sigmoid(-10) not being
    # exactly 0 on the already-correct target==0 cells).
    assert abs(boosted_off - base_off) < 1e-3


def test_update_dense_delay_from_batch_with_fn_penalty_runs():
    model = DistanceTableCNN(in_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch = [_dense_item()]
    loss = update_dense_delay_from_batch(
        model, optimizer, batch, pos_weight=0.1, fn_penalty=2.0
    )
    assert np.isfinite(loss)


def test_trivial_baseline_fn_penalty_increases_loss():
    """The 'always off-path' baseline is wrong on every on-path cell, so a
    higher fn_penalty must strictly increase its loss."""
    item = _dense_item()
    base = trivial_baseline_dense_loss([item], pos_weight=0.1, fn_penalty=1.0)
    boosted = trivial_baseline_dense_loss([item], pos_weight=0.1, fn_penalty=4.0)
    assert boosted > base
