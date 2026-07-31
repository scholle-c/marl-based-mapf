"""Supervised delay *regression* pipeline.

Continuous counterpart of SupervisedDelayPipeline: trains the model to regress a
dense, real-valued per-cell delay target (default: the CBS funnel — geodesic
hop-distance to the agent's CBS path) with a softplus head and MSE loss, instead
of a binary on/off-path segmentation with sigmoid + BCE.

Track A: train/val MSE + MAE/RMSE (does the model fit the funnel?).
Track B: LaCAM SOC with the trained model vs. plain baseline — unchanged, reused
         from the binary pipeline.
"""

from __future__ import annotations

import random

import torch
from loguru import logger

from marl_path.model import (
    update_dense_delay_regression_from_batch,
    eval_dense_delay_regression_loss,
    compute_regression_metrics,
    trivial_baseline_regression_loss,
)
from marl_path.delay_methods import get_delay_method
from marl_path.dataset import CbsDataset

from .lacam_eval import track_b_log_suffix
from .supervised_delay import SupervisedDelayPipeline, _stream


class SupervisedDelayRegressionPipeline(SupervisedDelayPipeline):
    """CBS-supervised delay training with a continuous (regression) target."""

    def run_model_training(self) -> None:
        train_dir, test_dir, has_test_data = self._resolve_dataset_dirs()
        delay_method = get_delay_method(
            getattr(self.args, "delay_method", "cbs_funnel")
        )
        loss_type = getattr(self.args, "regression_loss", "mse")

        dataset = CbsDataset(
            train_dir,
            extractor=self.extractor,
            device=self.device,
            delay_method=delay_method,
        )
        train_ds, val_ds, train_loader, _ = self._split_dataset(dataset)

        logger.info(
            "Starting dense supervised REGRESSION (delay_method={}, loss={}): "
            "train_dir={}, dataset={} instances ({} train / {} val), epochs={}, "
            "lr={}, device={}",
            type(delay_method).__name__,
            loss_type,
            train_dir,
            len(dataset),
            len(train_ds),
            len(val_ds),
            self.args.epochs,
            self.args.lr,
            self.device.type,
        )
        self._log_test_data_info(has_test_data, test_dir)

        baseline_train = trivial_baseline_regression_loss(_stream(train_ds), loss_type)
        baseline_val = trivial_baseline_regression_loss(_stream(val_ds), loss_type)
        logger.info(
            "  trivial 'predict per-item mean' baseline {}: train={:.4f}  val={:.4f}",
            loss_type.upper(),
            baseline_train,
            baseline_val,
        )

        test_dataset: CbsDataset | None = None
        if has_test_data:
            test_dataset = CbsDataset(
                test_dir,
                extractor=self.extractor,
                device=self.device,
                delay_method=delay_method,
            )

        eval_interval = getattr(self.args, "eval_interval", 1)
        seed_gen = random.Random(getattr(self.args, "seed", 0))
        penalty_scale = getattr(self.args, "penalty_scale", 1.0)

        for epoch in range(self.args.epochs):
            # ── Track A: training ──────────────────────────────────────────
            batch_losses: list[float] = []
            for batch in train_loader:
                loss = update_dense_delay_regression_from_batch(
                    self.model, self.optimizer, batch, loss_type=loss_type
                )
                batch_losses.append(loss)
            train_loss = sum(batch_losses) / len(batch_losses)

            val_loss = eval_dense_delay_regression_loss(
                self.model, _stream(val_ds), loss_type=loss_type
            )
            train_mae, train_rmse = compute_regression_metrics(
                self.model, _stream(train_ds)
            )
            val_mae, val_rmse = compute_regression_metrics(self.model, _stream(val_ds))

            test_metrics_text = ""
            extra: dict[str, float | None] = {
                "baseline_val": baseline_val,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
            if test_dataset is not None:
                test_loss = eval_dense_delay_regression_loss(
                    self.model, _stream(test_dataset), loss_type=loss_type
                )
                test_mae, test_rmse = compute_regression_metrics(
                    self.model, _stream(test_dataset)
                )
                extra.update(
                    {"test_loss": test_loss, "test_mae": test_mae, "test_rmse": test_rmse}
                )
                test_metrics_text = (
                    f"  test={test_loss:.4f} (mae={test_mae:.3f} rmse={test_rmse:.3f})"
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
                "Epoch {:3d}/{}: train={:.4f}  val={:.4f} (mae={:.3f} rmse={:.3f})  "
                "baseline_val={:.4f}{}{}",
                epoch + 1,
                self.args.epochs,
                train_loss,
                val_loss,
                val_mae,
                val_rmse,
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

            self.maybe_save_checkpoint(epoch)
