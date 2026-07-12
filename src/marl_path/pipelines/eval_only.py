"""Standalone LaCAM evaluation: trained model vs. vanilla baseline vs. CBS-optimal.

Loads a trained model checkpoint (--model-file) and runs the LaCAM comparison
(Track B) on a dataset's test/ split (--dataset-dir), without any training.
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
    """Evaluate a trained --model-file against vanilla LaCAM on --dataset-dir/test."""

    def __init__(self, args: argparse.Namespace):
        if getattr(args, "model_file", None) is None:
            raise ValueError("--model-file is required for eval_only mode.")
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
        logger.info(
            "Evaluating model={} on {} of {} test instances from {}",
            self.args.model_file,
            n_used,
            n_test,
            test_dir,
        )

        self._eval_summary = eval_test_instances(
            test_dir,
            model=self.model,
            device=self.device,
            extractor=self.extractor,
            time_limit_ms=self.args.time_limit_ms,
            flg_star=self.args.flg_star,
            seed=self.args.seed,
            penalty_scale=getattr(self.args, "penalty_scale", 1.0),
            limit=eval_limit,
        )
        logger.info("Eval result:{}", track_b_log_suffix(self._eval_summary))

    def store_results(self) -> None:
        if self.args.record_mode == 0 or self._eval_summary is None:
            return
        os.makedirs(self.args.output_dir, exist_ok=True)
        summary = self._eval_summary
        data = {
            "model_file": str(self.args.model_file),
            "dataset_dir": str(self.args.dataset_dir),
            "n_instances_used": len(summary.cbs_socs),
            "baseline_soc_mean": summary.baseline.mean,
            "baseline_soc_std": summary.baseline.std,
            "baseline_success_rate": summary.baseline.success_rate,
            "model_soc_mean": summary.model.mean,
            "model_soc_std": summary.model.std,
            "model_success_rate": summary.model.success_rate,
            "win_rate": summary.win_rate,
            "gap_closed": summary.gap_closed,
            "cbs_soc_mean": summary.cbs_mean,
        }
        out_path = os.path.join(self.args.output_dir, "eval_result.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info("Saved eval result to {}", out_path)
