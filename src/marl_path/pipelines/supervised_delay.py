"""Supervised delay imitation pipeline.

Trains DistanceTableCNN on CBS-optimal paths from a pre-generated CbsDataset.
Plain MSE loss — no reward weighting, no EMA, no running mean.

Track A: train/val MSE on the dataset (does the model fit the targets?).
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

from marl_path.pycam import LaCAM
from marl_path.model import DelayBatchItem, get_soc, update_delay_from_batch, eval_delay_loss
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
    win_rate: float         # fraction of runs where model SOC < baseline SOC
    gap_closed: float | None   # (baseline_mean - model_mean) / (baseline_mean - cbs_mean)
    cbs_mean: float | None     # mean CBS-optimal SOC across test instances
    cbs_socs: list[float]      # per-instance CBS-optimal SOCs


class SupervisedDelayPipeline(DefaultTrainingPipeline):
    """CBS-supervised delay training pipeline."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def run_model_training(self) -> None:
        dataset_dir = Path(self.args.dataset_dir)
        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"

        # Fall back to dataset_dir itself when no train/ subdir exists
        if not train_dir.exists():
            logger.warning(
                "No train/ subdir found in {}; loading all instances from root.",
                dataset_dir,
            )
            train_dir = dataset_dir

        dataset = CbsDataset(train_dir, extractor=self.extractor, device=self.device)

        val_n = max(1, int(len(dataset) * self.args.val_split))
        train_n = len(dataset) - val_n
        generator = torch.Generator().manual_seed(getattr(self.args, "seed_training", 0) or 0)
        train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=generator)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            generator=generator,
        )

        has_test_data = test_dir.exists() and any(test_dir.glob("*.npz"))

        logger.info(
            "Starting supervised training: train_dir={}, dataset={} instances "
            "({} train / {} val), epochs={}, lr={}, device={}",
            train_dir,
            len(dataset),
            train_n,
            val_n,
            self.args.epochs,
            self.args.lr,
            self.device.type,
        )
        if has_test_data:
            n_test = sum(1 for _ in test_dir.glob("*.npz"))
            logger.info("LaCAM eval will use {} test instances from {}", n_test, test_dir)
        else:
            logger.info("No test/ data found; Track B (LaCAM eval) will be skipped.")

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

            val_batch = cast(list[DelayBatchItem], [val_ds[i] for i in range(len(val_ds))])
            val_loss = eval_delay_loss(self.model, val_batch)
            mean_pred = _mean_predicted_delay(self.model, val_batch)

            # ── Track B: LaCAM SOC eval on test instances ──────────────────
            soc_model: float | None = None
            eval_summary: LacamComparisonSummary | None = None
            if has_test_data and (epoch + 1) % eval_interval == 0:
                eval_seed = seed_gen.randint(0, SEED_MAX)
                eval_summary = _eval_test_instances(
                    test_dir,
                    model=self.model,
                    device=self.device,
                    extractor=self.extractor,
                    time_limit_ms=self.args.time_limit_ms,
                    flg_star=self.args.flg_star,
                    seed=eval_seed,
                )
                soc_model = eval_summary.model.mean

            self.training_stats.record_epoch(train_loss, soc=soc_model, val_loss=val_loss)

            if eval_summary is not None:
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
                logger.info(
                    "Epoch {:3d}/{}: train={:.4f}  val={:.4f}  pred_delay={:.3f}  "
                    "soc_model={:.1f}±{:.1f} [min={:.0f} max={:.0f}]  "
                    "soc_baseline={:.1f}±{:.1f} [min={:.0f} max={:.0f}]  "
                    "win_rate={:.1f}%{}{}",
                    epoch + 1,
                    self.args.epochs,
                    train_loss,
                    val_loss,
                    mean_pred,
                    m.mean, m.std, m.min, m.max,
                    b.mean, b.std, b.min, b.max,
                    100.0 * eval_summary.win_rate,
                    cbs_text,
                    gap_text,
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
        split_name, len(items), pct, mean_delay,
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

        b_soc = _run_lacam_once(grid, starts, goals, seed, time_limit_ms=time_limit_ms, flg_star=flg_star)
        m_soc = _run_lacam_once(
            grid, starts, goals, seed,
            model=model, device=device, extractor=extractor,
            time_limit_ms=time_limit_ms, flg_star=flg_star,
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
        gap_closed=_optimality_gap_closed(baseline_summary.mean, model_summary.mean, cbs_mean),
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
