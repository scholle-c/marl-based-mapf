"""Supervised delay imitation pipeline.

Trains DistanceTableCNN on CBS-optimal paths from a pre-generated CbsDataset.

Two delay-target modes (--delay-target):
  first_visit          (default) — sparse, per-path-cell regression targets
                        (FirstVisitDelay), softplus head, MSE loss. The
                        original pipeline, unchanged.
  non_optimal_penalty  — dense, full-grid binary segmentation targets
                        (NonOptimalPenaltyDelay), sigmoid head, BCE loss.

Track A: train/val loss on the dataset (does the model fit the targets?).
Track B: SOC when LaCAM runs with the trained model vs. plain LaCAM baseline.
         Evaluated on test instances from dataset_dir/test/ (skipped if absent).
         CBS optimal SOC is read directly from the cached paths — no re-running CBS.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from loguru import logger

import marl_path.constants as consts
from marl_path.pycam import LaCAM
from marl_path.model import (
    DelayBatchItem,
    DenseDelayBatchItem,
    get_soc,
    update_delay_from_batch,
    eval_delay_loss,
    update_dense_delay_from_batch,
    eval_dense_delay_loss,
    trivial_baseline_dense_loss,
    compute_mask_iou_f1,
    compute_cell_overlap,
)
from marl_path.delay_methods import get_delay_method
from marl_path.dataset import CbsDataset
from marl_path.dataset.instance import CachedInstance
from marl_path.shared.mapf_utils import Config, get_grid, get_scenario

from .base import DefaultTrainingPipeline

SEED_MAX = 2**32 - 1


@dataclass
class LacamEvalSummary:
    """Summary statistics for a LaCAM evaluation across instances/seeds."""

    socs: list[float]
    mean: float
    std: float
    min: float
    max: float
    success_rate: float


@dataclass
class LacamComparisonSummary:
    """Comparison between baseline, model, and CBS-optimal SOCs."""

    baseline: LacamEvalSummary
    model: LacamEvalSummary
    win_rate: float  # fraction of runs where model SOC < baseline SOC
    gap_closed: (
        float | None
    )  # (baseline_mean - model_mean) / (baseline_mean - cbs_mean)
    cbs_mean: float | None  # mean CBS-optimal SOC across test instances
    cbs_socs: list[float]  # per-instance CBS-optimal SOCs


class SupervisedDelayPipeline(DefaultTrainingPipeline):
    """CBS-supervised delay training pipeline."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def run_model_training(self) -> None:
        delay_target = getattr(
            self.args, "delay_target", consts.DELAY_TARGET_FIRST_VISIT
        )
        if delay_target == consts.DELAY_TARGET_NON_OPTIMAL_PENALTY:
            self._run_dense_training()
        else:
            self._run_sparse_training()

    # ------------------------------------------------------------------ #
    # first_visit: sparse targets, softplus head, MSE loss (original)      #
    # ------------------------------------------------------------------ #

    def _run_sparse_training(self) -> None:
        train_dir, test_dir, has_test_data = self._resolve_dataset_dirs()
        dataset = CbsDataset(train_dir, extractor=self.extractor, device=self.device)

        train_ds, val_ds, train_loader, generator = self._split_dataset(dataset)

        logger.info(
            "Starting supervised training (first_visit): train_dir={}, dataset={} instances "
            "({} train / {} val), epochs={}, lr={}, device={}",
            train_dir,
            len(dataset),
            len(train_ds),
            len(val_ds),
            self.args.epochs,
            self.args.lr,
            self.device.type,
        )
        self._log_test_data_info(has_test_data, test_dir)

        _log_delay_stats("train", [train_ds[i] for i in range(len(train_ds))])  # type: ignore[arg-type]
        _log_delay_stats("val  ", [val_ds[i] for i in range(len(val_ds))])  # type: ignore[arg-type]

        eval_interval = getattr(self.args, "eval_interval", 1)
        seed_gen = random.Random(getattr(self.args, "seed", 0))

        for epoch in range(self.args.epochs):
            # ── Track A: training ──────────────────────────────────────────
            batch_losses: list[float] = []
            for batch in train_loader:
                loss = update_delay_from_batch(self.model, self.optimizer, batch)
                batch_losses.append(loss)
            train_loss = sum(batch_losses) / len(batch_losses)

            val_batch = cast(
                list[DelayBatchItem], [val_ds[i] for i in range(len(val_ds))]
            )
            val_loss = eval_delay_loss(self.model, val_batch)
            mean_pred = _mean_predicted_delay(self.model, val_batch)

            # ── Track B: LaCAM SOC eval on test instances ──────────────────
            soc_model, eval_summary = self._maybe_eval_track_b(
                has_test_data, test_dir, epoch, eval_interval, seed_gen
            )
            self.training_stats.record_epoch(
                train_loss, soc=soc_model, val_loss=val_loss
            )

            if eval_summary is not None:
                logger.info(
                    "Epoch {:3d}/{}: train={:.4f}  val={:.4f}  pred_delay={:.3f}{}",
                    epoch + 1,
                    self.args.epochs,
                    train_loss,
                    val_loss,
                    mean_pred,
                    _track_b_log_suffix(eval_summary),
                )
            else:
                logger.info(
                    "Epoch {:3d}/{}: train={:.4f}  val={:.4f}  pred_delay={:.3f}",
                    epoch + 1,
                    self.args.epochs,
                    train_loss,
                    val_loss,
                    mean_pred,
                )

    # ------------------------------------------------------------------ #
    # non_optimal_penalty: dense targets, sigmoid head, BCE loss            #
    # ------------------------------------------------------------------ #

    def _run_dense_training(self) -> None:
        train_dir, test_dir, has_test_data = self._resolve_dataset_dirs()
        delay_method = get_delay_method(
            getattr(self.args, "delay_method", "non_optimal_penalty")
        )
        pos_weight = getattr(self.args, "pos_weight", 0.05)

        dataset = CbsDataset(
            train_dir,
            extractor=self.extractor,
            device=self.device,
            mode="dense",
            delay_method=delay_method,
        )
        train_ds, val_ds, train_loader, generator = self._split_dataset(dataset)
        train_batch = cast(
            list[DenseDelayBatchItem], [train_ds[i] for i in range(len(train_ds))]
        )
        val_batch = cast(
            list[DenseDelayBatchItem], [val_ds[i] for i in range(len(val_ds))]
        )

        logger.info(
            "Starting dense supervised training (non_optimal_penalty): train_dir={}, "
            "dataset={} instances ({} train / {} val), epochs={}, lr={}, pos_weight={}, "
            "device={}",
            train_dir,
            len(dataset),
            len(train_ds),
            len(val_ds),
            self.args.epochs,
            self.args.lr,
            pos_weight,
            self.device.type,
        )
        self._log_test_data_info(has_test_data, test_dir)

        baseline_train = trivial_baseline_dense_loss(train_batch, pos_weight)
        baseline_val = trivial_baseline_dense_loss(val_batch, pos_weight)
        logger.info(
            "  trivial 'always off-path' baseline BCE: train={:.4f}  val={:.4f}",
            baseline_train,
            baseline_val,
        )

        test_batch: list[DenseDelayBatchItem] | None = None
        if has_test_data:
            test_dataset = CbsDataset(
                test_dir,
                extractor=self.extractor,
                device=self.device,
                mode="dense",
                delay_method=delay_method,
            )
            test_batch = cast(
                list[DenseDelayBatchItem],
                [test_dataset[i] for i in range(len(test_dataset))],
            )
            overlap = compute_cell_overlap(train_batch, test_batch)
            logger.info(
                "  train/test on-path cell overlap: {:.1f}% "
                "(high overlap + high test IoU/F1 suggests memorization, not generalization)",
                100.0 * overlap,
            )

        eval_interval = getattr(self.args, "eval_interval", 1)
        seed_gen = random.Random(getattr(self.args, "seed", 0))
        penalty_scale = getattr(self.args, "penalty_scale", 1.0)

        for epoch in range(self.args.epochs):
            # ── Track A: training ──────────────────────────────────────────
            batch_losses: list[float] = []
            for batch in train_loader:
                loss = update_dense_delay_from_batch(
                    self.model, self.optimizer, batch, pos_weight=pos_weight
                )
                batch_losses.append(loss)
            train_loss = sum(batch_losses) / len(batch_losses)

            val_loss = eval_dense_delay_loss(
                self.model, val_batch, pos_weight=pos_weight
            )
            train_iou, train_f1 = compute_mask_iou_f1(self.model, train_batch)
            val_iou, val_f1 = compute_mask_iou_f1(self.model, val_batch)

            test_metrics_text = ""
            extra: dict[str, float | None] = {
                "baseline_bce_val": baseline_val,
                "train_iou": train_iou,
                "train_f1": train_f1,
                "val_iou": val_iou,
                "val_f1": val_f1,
            }
            if test_batch is not None:
                test_loss = eval_dense_delay_loss(
                    self.model, test_batch, pos_weight=pos_weight
                )
                test_iou, test_f1 = compute_mask_iou_f1(self.model, test_batch)
                extra.update(
                    {"test_bce": test_loss, "test_iou": test_iou, "test_f1": test_f1}
                )
                test_metrics_text = (
                    f"  test={test_loss:.4f} (iou={test_iou:.3f} f1={test_f1:.3f})"
                )

            # ── Track B: LaCAM SOC eval on test instances ──────────────────
            soc_model, eval_summary = self._maybe_eval_track_b(
                has_test_data,
                test_dir,
                epoch,
                eval_interval,
                seed_gen,
                penalty_scale=penalty_scale,
            )
            self.training_stats.record_epoch(
                train_loss, soc=soc_model, val_loss=val_loss, extra=extra
            )

            logger.info(
                "Epoch {:3d}/{}: train={:.4f}  val={:.4f} (iou={:.3f} f1={:.3f})  "
                "baseline_val={:.4f}{}{}",
                epoch + 1,
                self.args.epochs,
                train_loss,
                val_loss,
                val_iou,
                val_f1,
                baseline_val,
                test_metrics_text,
                _track_b_log_suffix(eval_summary) if eval_summary is not None else "",
            )

            lr_before = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            lr_after = self.optimizer.param_groups[0]["lr"]
            if lr_after < lr_before:
                logger.info(
                    "  LR reduced: {:.2e} -> {:.2e} (val loss plateaued)",
                    lr_before,
                    lr_after,
                )

    # ------------------------------------------------------------------ #
    # Shared helpers                                                        #
    # ------------------------------------------------------------------ #

    def _resolve_dataset_dirs(self) -> tuple[Path, Path, bool]:
        dataset_dir = Path(self.args.dataset_dir)
        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"
        if not train_dir.exists():
            logger.warning(
                "No train/ subdir found in {}; loading all instances from root.",
                dataset_dir,
            )
            train_dir = dataset_dir
        has_test_data = test_dir.exists() and any(test_dir.glob("*.npz"))
        return train_dir, test_dir, has_test_data

    def _log_test_data_info(self, has_test_data: bool, test_dir: Path) -> None:
        if has_test_data:
            n_test = sum(1 for _ in test_dir.glob("*.npz"))
            logger.info(
                "LaCAM eval will use {} test instances from {}", n_test, test_dir
            )
        else:
            logger.info("No test/ data found; Track B (LaCAM eval) will be skipped.")

    def _split_dataset(self, dataset: CbsDataset):
        val_n = max(1, int(len(dataset) * self.args.val_split))
        train_n = len(dataset) - val_n
        generator = torch.Generator().manual_seed(
            getattr(self.args, "seed_training", 0) or 0
        )
        train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=generator)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            generator=generator,
        )
        return train_ds, val_ds, train_loader, generator

    def _maybe_eval_track_b(
        self,
        has_test_data: bool,
        test_dir: Path,
        epoch: int,
        eval_interval: int,
        seed_gen: random.Random,
        penalty_scale: float = 1.0,
    ) -> tuple[float | None, LacamComparisonSummary | None]:
        if not (has_test_data and (epoch + 1) % eval_interval == 0):
            return None, None
        eval_seed = seed_gen.randint(0, SEED_MAX)
        eval_summary = _eval_test_instances(
            test_dir,
            model=self.model,
            device=self.device,
            extractor=self.extractor,
            time_limit_ms=self.args.time_limit_ms,
            flg_star=self.args.flg_star,
            seed=eval_seed,
            penalty_scale=penalty_scale,
        )
        return eval_summary.model.mean, eval_summary


def _track_b_log_suffix(eval_summary: LacamComparisonSummary) -> str:
    b = eval_summary.baseline
    m = eval_summary.model
    cbs_text = (
        f"  soc_cbs={eval_summary.cbs_mean:.1f}"
        if eval_summary.cbs_mean is not None
        else ""
    )
    gap_text = (
        f"  gap_closed={100.0 * eval_summary.gap_closed:.1f}%"
        if eval_summary.gap_closed is not None
        else ""
    )
    return (
        f"  soc_model={m.mean:.1f}±{m.std:.1f} [min={m.min:.0f} max={m.max:.0f}]  "
        f"soc_baseline={b.mean:.1f}±{b.std:.1f} [min={b.min:.0f} max={b.max:.0f}]  "
        f"win_rate={100.0 * eval_summary.win_rate:.1f}%{cbs_text}{gap_text}"
    )


def _log_delay_stats(split_name: str, items: list) -> None:
    """Log mean target delay and % non-zero delay cells for a split."""
    total_cells = 0
    nonzero_cells = 0
    sum_delay = 0.0
    for item in items:
        targets = item.targets.cpu().numpy()
        bfs_flat = np.concatenate([b for b in item.bfs_distances])
        delays = targets - bfs_flat
        total_cells += len(delays)
        nonzero_cells += int((delays > 1e-6).sum())
        sum_delay += float(delays.mean())
    pct = 100.0 * nonzero_cells / total_cells if total_cells else 0.0
    mean_delay = sum_delay / len(items) if items else 0.0
    logger.info(
        "  {} split: {} instances, {:.1f}% non-zero delay cells, mean delay per instance={:.3f}",
        split_name,
        len(items),
        pct,
        mean_delay,
    )


def _mean_predicted_delay(model: Any, batch: list) -> float:
    """Mean predicted delay (across all agents and path cells) over a batch."""
    from marl_path.model.training import _batch_delay_tables, _get_via_coordinates

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for item in batch:
            delay_tables = _batch_delay_tables(model, item.input_tensors)
            for agent_idx, path in enumerate(item.paths):
                preds = _get_via_coordinates(delay_tables[agent_idx], path)
                total += float(preds.mean())
                count += 1
    return total / count if count else 0.0


def _cbs_soc_from_cached_paths(paths: list[list]) -> float:
    """Compute SOC from CBS paths stored in agent-first format (paths[agent][t])."""
    if not paths:
        return 0.0
    max_len = max(len(p) for p in paths)
    padded = [p + [p[-1]] * (max_len - len(p)) for p in paths]
    solution = [[padded[a][t] for a in range(len(padded))] for t in range(max_len)]
    return float(get_soc(solution))


def _eval_test_instances(
    test_dir: Path,
    model=None,
    device=None,
    extractor=None,
    time_limit_ms: int = 3000,
    flg_star: bool = True,
    seed: int = 0,
    penalty_scale: float = 1.0,
) -> LacamComparisonSummary:
    """Evaluate LaCAM (baseline vs model) on all cached test instances.

    Each test instance is run once with the given seed. CBS-optimal SOC is
    read directly from the cached paths — no CBS re-execution needed.
    """
    baseline_socs: list[float] = []
    model_socs: list[float] = []
    cbs_socs: list[float] = []
    n_instances = 0

    for npz_path in sorted(test_dir.glob("*.npz")):
        n_instances += 1
        instance = CachedInstance.load(npz_path)

        grid = get_grid(instance.map_file)
        all_starts, all_goals = get_scenario(instance.scen_file)
        starts = Config(positions=[all_starts[i] for i in instance.agent_indices])
        goals = Config(positions=[all_goals[i] for i in instance.agent_indices])

        cbs_soc = _cbs_soc_from_cached_paths(instance.paths)
        cbs_socs.append(cbs_soc)

        b_soc = _run_lacam_once(
            grid, starts, goals, seed, time_limit_ms=time_limit_ms, flg_star=flg_star
        )
        m_soc = _run_lacam_once(
            grid,
            starts,
            goals,
            seed,
            model=model,
            device=device,
            extractor=extractor,
            time_limit_ms=time_limit_ms,
            flg_star=flg_star,
            penalty_scale=penalty_scale,
        )

        if b_soc is not None:
            baseline_socs.append(b_soc)
        if m_soc is not None:
            model_socs.append(m_soc)

    baseline_summary = _summarize_socs(baseline_socs, n_instances)
    model_summary = _summarize_socs(model_socs, n_instances)

    cbs_mean: float | None = float(np.mean(cbs_socs)) if cbs_socs else None

    return LacamComparisonSummary(
        baseline=baseline_summary,
        model=model_summary,
        win_rate=_paired_win_rate(model_socs, baseline_socs),
        gap_closed=_optimality_gap_closed(
            baseline_summary.mean, model_summary.mean, cbs_mean
        ),
        cbs_mean=cbs_mean,
        cbs_socs=cbs_socs,
    )


def _run_lacam_once(
    grid,
    starts,
    goals,
    seed: int,
    model=None,
    device=None,
    extractor=None,
    time_limit_ms: int = 3000,
    flg_star: bool = True,
    penalty_scale: float = 1.0,
) -> float | None:
    """Run LaCAM once for a single seed; return SOC or None on failure."""
    planner = LaCAM()
    solution = planner.solve(
        grid=grid,
        starts=starts,
        goals=goals,
        seed=seed,
        model=model,
        device=device,
        extractor=extractor,
        time_limit_ms=time_limit_ms,
        flg_star=flg_star,
        verbose=0,
        penalty_scale=penalty_scale,
    )
    return float(get_soc(solution)) if solution else None


def _summarize_socs(socs: list[float], num_total: int) -> LacamEvalSummary:
    if socs:
        mean = float(np.mean(socs))
        std = float(np.std(socs))
        min_soc = float(np.min(socs))
        max_soc = float(np.max(socs))
    else:
        mean = float("nan")
        std = float("nan")
        min_soc = float("nan")
        max_soc = float("nan")
    success_rate = len(socs) / num_total if num_total else float("nan")
    return LacamEvalSummary(
        socs=socs,
        mean=mean,
        std=std,
        min=min_soc,
        max=max_soc,
        success_rate=success_rate,
    )


def _paired_win_rate(model_socs: list[float], baseline_socs: list[float]) -> float:
    paired = [m < b for m, b in zip(model_socs, baseline_socs)]
    return float(np.mean(paired)) if paired else float("nan")


def _optimality_gap_closed(
    baseline_soc: float,
    model_soc: float,
    cbs_soc: float | None,
) -> float | None:
    if cbs_soc is None:
        return None
    denom = baseline_soc - cbs_soc
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return None
    return (baseline_soc - model_soc) / denom
