# Task: Adapt training pipeline to learn `NonOptimalPenaltyDelay`

## Context

We validated that `NonOptimalPenaltyDelay` (see `delay_methods.py`) is a strong
*oracle* signal for improving LaCAM's SOC — but it currently requires CBS at
inference time (it's computed directly from CBS paths, not predicted by a
model). The goal now is to train `DistanceTableCNN` (or a variant) to
approximate this target, so CBS is no longer needed at inference.

**Important:** this target is structurally different from the existing
`FirstVisitDelay` target that `SupervisedDelayPipeline` was originally built
for. Do not assume the existing sparse, path-only training loop is a good fit
— see "Why the current loop doesn't fit" below.

## What the target actually looks like

```python
class NonOptimalPenaltyDelay(DelayMethod):
    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        path = paths[agent_idx]
        delay_map = np.ones(grid.shape, dtype=np.float32)
        for coord in path:
            delay_map[coord] = 0.0
        return delay_map
```

This is a **dense, binary, per-cell segmentation mask** over the *entire*
grid (not just path cells): 0 for cells on the agent's CBS-optimal path, 1
everywhere else. It depends on the full multi-agent configuration (other
agents' starts/goals influence which path CBS assigns), not just the single
agent's own geometry.

Two structural properties matter a lot for training:

1. **Extreme class imbalance.** On a typical instance, path cells are a small
   minority of all free cells (e.g. ~40 path cells out of ~1000 free cells on
   `empty-32-32`). A model that predicts ~1 everywhere will already get a
   deceptively low MSE/BCE. Any reported "it's learning" claim must be backed
   by a class-imbalance-robust metric (see Acceptance Criteria).
2. **Label degeneracy on symmetric maps.** Grids admit many equal-cost
   shortest paths between two points; CBS/EECBS picks one somewhat arbitrarily
   via internal tie-breaking when there's no conflict. All other equally-good
   alternative paths still get labeled 1. This introduces label noise that no
   input feature can resolve — expect the achievable ceiling on very open
   maps (e.g. `empty-32-32`) to reflect this, and don't chase 100% mask
   accuracy there.

## Why the current training loop doesn't fit (as-is)

In `training.py`:
- `prepare_delay_batch_item` builds targets via `_get_path_target_first_visit`
  (fv-dist along the path) — this is specific to `FirstVisitDelay` and is
  **not** what we want for `NonOptimalPenaltyDelay`.
- `update_delay_from_batch` extracts predictions only at path coordinates via
  `_get_via_coordinates(delay_tables[agent_idx], path)` — i.e. loss is only
  computed on path cells. `NonOptimalPenaltyDelay` gives us a label for every
  cell in the grid; throwing that away and only supervising path cells
  recreates the old sparse-supervision problem (no gradient signal for cells
  never visited in training instances), which is very likely why earlier
  CBS-supervised training didn't generalize.
- `DistanceTableCNN`'s output head is `softplus(x) * DELAY_SCALE` (`DELAY_SCALE
  = 0.1`) — built for small positive continuous residuals, not a binary
  target. Reaching an output near `1.0` requires large pre-activation values,
  which is an awkward, indirect way to represent a step function.

## Required changes

### 1. New target construction path (dense, full-grid)
Add a way to build training targets as the **full `grid.shape` array**
returned by `NonOptimalPenaltyDelay.compute(...)`, not the sparse per-path-cell
list used for `FirstVisitDelay`. This likely means a new/parallel function
alongside `prepare_delay_batch_item` (e.g. `prepare_dense_delay_batch_item`)
that stores a `(H, W)` (or flattened) target per agent instead of a
per-path-cell list. Keep the existing sparse path unchanged so
`FirstVisitDelay` training still works — this should be an additive branch,
not a rewrite of the existing pipeline (see `SupervisedDelayPipeline` —
extend, don't replace).

### 2. Dense loss over the whole grid
Modify (or add a variant of) `update_delay_from_batch` / `eval_delay_loss` so
that, in dense mode, the loss is computed over **all free cells in the grid**,
not just path cells. Obstacle cells should be masked out of the loss (they're
never queried by `DistTable` anyway).

### 3. Output activation + loss function for this target
For the dense/binary target:
- Change the final activation to `sigmoid` instead of `softplus(x) *
  DELAY_SCALE` (only for this training mode — don't break the existing
  softplus head used elsewhere; consider a `DistanceTableCNN` constructor
  flag, e.g. `output_activation: Literal["softplus", "sigmoid"]`, or a small
  subclass).
- Use `BCEWithLogitsLoss` instead of `mse_loss` for this target. Since the
  positive class (delay=1, i.e. "not on path") vastly outnumbers the negative
  class (delay=0, "on path"), pass `pos_weight < 1` (or equivalently weight
  the 0-class up) to `BCEWithLogitsLoss` so the loss isn't dominated by the
  majority class. Make this weight a config/CLI parameter — it'll need
  tuning.
- At inference (`DistTable.compute_delay_model`), the sigmoid output is in
  `[0, 1]`; the actual penalty magnitude added to `h_bfs` should be a
  separate, explicit scale hyperparameter (e.g. `delay = sigmoid_output *
  penalty_scale`), not baked into the activation like `DELAY_SCALE` currently
  is. Expose this as a config value so it can be swept independently of model
  training.

### 4. Enrich the "other agents" input features
The target fundamentally depends on other agents' configurations (a cell is
only "non-optimal" because of interaction with others). Check
`feature_extraction.py` — the existing `OtherAgentsExtractor` reportedly only
encodes other agents' **goals**. For this target, extend it to also encode
other agents' **start positions** (and/or their own BFS distance fields,
stacked or aggregated e.g. via min/mean across agents), so the model has a
chance to explain why a specific alternative path was excluded.

### 5. Baseline + evaluation metrics beyond loss
Add to the training loop / eval:
- **Trivial baseline log**: log the loss (both BCE and the eventual downstream
  SOC/gap_closed metric) of a constant "always predict 1" model, so absolute
  loss numbers are interpretable against the class-imbalance floor.
- **Mask IoU / F1**: on both train and held-out test instances, threshold the
  sigmoid output (e.g. at 0.5) and compute IoU or F1 between the predicted
  "on-path" mask and the true CBS path mask. This is the metric that actually
  tells us whether the model learned something beyond majority-class
  prediction — don't rely on BCE/MSE decrease alone.
- Keep existing Track B (`_eval_test_instances`, `gap_closed`, `win_rate`)
  as-is — that's still the ultimate downstream metric and doesn't need
  changes.

### 6. Train/test split for generalization testing
For the initial learnability experiment (map: `empty-32-32`, few agents),
generate train and test instances with **disjoint agent start/goal
configurations**, and additionally log how much overlap exists between cells
visited in train vs. test instances. If test performance correlates strongly
with cell-overlap, that's a sign of memorization rather than generalization —
worth surfacing as a diagnostic, not just a side note.

## Files likely touched
- `delay_methods.py` — no change needed, `NonOptimalPenaltyDelay` already
  returns the right dense array.
- `src/marl_path/model/training.py` — new dense batch-item + loss path.
- `src/marl_path/model/definition.py` — configurable output activation.
- `src/marl_path/model/feature_extraction.py` — enrich other-agent encoding.
- `src/marl_path/pipelines/supervised_delay.py` — wire up dense mode as a
  config/CLI option; add IoU/F1 + trivial-baseline logging to the epoch log
  line.
- `src/marl_path/dataset/generate.py` — check whether dataset caching needs
  to store the full dense target array per agent, or whether it can be
  recomputed on the fly from cached CBS paths (`CachedInstance`) at load
  time — prefer recomputing on the fly if cheap, to avoid dataset format
  churn.

## Explicitly out of scope for this task
- Increasing the model's receptive field (dilated convs / U-Net-style
  encoder-decoder) — flagged as a good idea for later once dense-target
  training is confirmed to work at all, but don't do it preemptively.
- Softening the degenerate-tie-break label noise (e.g. via diffusion or
  multi-optimal-path detection) — parked as a future refinement if plain
  binary training turns out too noisy.
- Any change to the LaCAM/PIBT solver integration.

## Acceptance criteria
1. Training runs in dense mode on `empty-32-32` with a handful of agents
   without errors, logging: BCE loss, trivial-baseline BCE loss, mask IoU/F1
   for train and test splits, and (via existing Track B) SOC/gap_closed/
   win_rate vs. baseline and CBS-optimal.
2. Mask IoU/F1 on the test split is meaningfully above what a trivial
   "always predict background class" baseline would achieve.
3. The old `FirstVisitDelay`/sparse/softplus training path still runs
   unchanged (regression check — this task should be additive).

## Open questions to flag back to Carsten (don't guess silently)
- Exact `pos_weight` / class-balance handling for `BCEWithLogitsLoss` —
  needs empirical tuning, start with inverse class frequency as a first
  guess.
- Whether dense targets should be cached in the `.npz` instance files
  (bigger storage, faster loading) or recomputed on the fly from paths at
  train time (simpler, avoids re-generating cached datasets).
- Naming for the new dense/sigmoid training mode (CLI flag, config key,
  pipeline class name) — pick something consistent with existing
  `--delay-method` / `--pipeline-mode` conventions.
