"""Standalone LaCAM evaluation: vanilla baseline vs. CBS-optimal, optionally vs. a trained model.

Runs the LaCAM comparison (Track B) on a dataset's test/ split
(--dataset-dir), without any training. If --model-file is given, a trained
model checkpoint is also evaluated against the baseline; otherwise only the
vanilla-LaCAM-vs-CBS-optimal gap is reported, which is useful to check
upfront how much headroom a dataset actually has before training a model.
Decoupled from SupervisedDelayPipeline's epoch loop so it can be pointed at
however many test instances make sense via --eval-limit (default: all),
instead of being bound to whatever eval_interval happened to be during
training.
"""

from __future__ import annotations

import argparse
import json
import os

from loguru import logger

from .base import DefaultTrainingPipeline
from .lacam_eval import (
    LacamComparisonSummary,
    eval_test_instances,
    resolve_test_dir,
    track_b_log_suffix,
)


class EvalOnlyPipeline(DefaultTrainingPipeline):
    """Evaluate vanilla LaCAM vs. CBS-optimal on --dataset-dir/test, optionally vs. --model-file."""

    def __init__(self, args: argparse.Namespace):
        if getattr(args, "dataset_dir", None) is None:
            raise ValueError("--dataset-dir is required for eval_only mode.")
        super().__init__(args)
        self._eval_summary: LacamComparisonSummary | None = None

    def run_model_training(self) -> None:
        test_dir, has_test_data = resolve_test_dir(self.args.dataset_dir)
        if not has_test_data:
            raise ValueError(f"No test/ data found in {self.args.dataset_dir}")

        n_test = sum(1 for _ in test_dir.glob("*.npz"))
        eval_limit = getattr(self.args, "eval_limit", None)
        n_used = min(eval_limit, n_test) if eval_limit else n_test
        has_model = self.args.model_file is not None
        logger.info(
            "Evaluating {} on {} of {} test instances from {}",
            f"model={self.args.model_file}"
            if has_model
            else "vanilla LaCAM baseline only (no --model-file given)",
            n_used,
            n_test,
            test_dir,
        )

        self._eval_summary = eval_test_instances(
            test_dir,
            model=self.model if has_model else None,
            device=self.device,
            extractor=self.extractor,
            time_limit_ms=self.args.time_limit_ms,
            flg_star=self.args.flg_star,
            seed=self.args.seed,
            penalty_scale=getattr(self.args, "penalty_scale", 1.0),
            limit=eval_limit,
            num_seeds=getattr(self.args, "eval_seeds", 1) or 1,
        )
        logger.info("Eval result:{}", track_b_log_suffix(self._eval_summary))

    def store_results(self) -> None:
        if self.args.record_mode == 0 or self._eval_summary is None:
            return
        os.makedirs(self.args.output_dir, exist_ok=True)
        summary = self._eval_summary
        data = {
            "model_file": str(self.args.model_file) if self.args.model_file else None,
            "dataset_dir": str(self.args.dataset_dir),
            "n_instances_used": len(summary.cbs_socs),
            "baseline_soc_mean": summary.baseline.mean,
            "baseline_soc_std": summary.baseline.std,
            "baseline_success_rate": summary.baseline.success_rate,
            "model_soc_mean": summary.model.mean if summary.model else None,
            "model_soc_std": summary.model.std if summary.model else None,
            "model_success_rate": summary.model.success_rate if summary.model else None,
            "win_rate": summary.win_rate,
            "gap_closed": summary.gap_closed,
            "cbs_soc_mean": summary.cbs_mean,
        }
        out_path = os.path.join(self.args.output_dir, "eval_result.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info("Saved eval result to {}", out_path)
