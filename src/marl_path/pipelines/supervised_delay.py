"""Supervised delay imitation pipeline.

Trains DistanceTableCNN on CBS-optimal paths from a pre-generated CbsDataset,
using dense, full-grid binary segmentation targets (default: NonOptimalPenaltyDelay,
selectable via --delay-method), sigmoid head, BCE loss.

Track A: train/val loss on the dataset (does the model fit the targets?).
Track B: SOC when LaCAM runs with the trained model vs. plain LaCAM baseline.
         Evaluated on test instances from dataset_dir/test/ (skipped if absent).
         CBS optimal SOC is read directly from the cached paths — no re-running CBS.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader, random_split
from loguru import logger

from marl_path.model import (
    DenseDelayBatchItem,
    update_dense_delay_from_batch,
    eval_dense_delay_loss,
    trivial_baseline_dense_loss,
    compute_mask_iou_f1,
    compute_cell_overlap,
)
from marl_path.delay_methods import get_delay_method
from marl_path.dataset import CbsDataset

from .base import DefaultTrainingPipeline
from .lacam_eval import (
    LacamComparisonSummary,
    eval_test_instances,
    track_b_log_suffix,
)

SEED_MAX = 2**32 - 1


class SupervisedDelayPipeline(DefaultTrainingPipeline):
    """CBS-supervised delay training pipeline."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def run_model_training(self) -> None:
        train_dir, test_dir, has_test_data = self._resolve_dataset_dirs()
        delay_method = get_delay_method(
            getattr(self.args, "delay_method", "non_optimal_penalty")
        )
        pos_weight = getattr(self.args, "pos_weight", 0.05)

        dataset = CbsDataset(
            train_dir,
            extractor=self.extractor,
            device=self.device,
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
            "Starting dense supervised training (delay_method={}): train_dir={}, "
            "dataset={} instances ({} train / {} val), epochs={}, lr={}, pos_weight={}, "
            "device={}",
            type(delay_method).__name__,
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
                track_b_log_suffix(eval_summary) if eval_summary is not None else "",
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
        eval_summary = eval_test_instances(
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
