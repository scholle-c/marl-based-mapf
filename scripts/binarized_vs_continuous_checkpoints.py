"""Continuous vs. binarized model delay output across training checkpoints.

Throwaway experiment script, follow-up to scripts/oracle_noise_sweep.py's
--round sanity check: rounding the *exact* oracle target to {0,1} trivially
undoes small continuous noise there (target is binary, tested magnitudes are
all < 0.5). This checks whether the analogous idea -- hard-thresholding the
*model's* continuous sigmoid output at 0.5 instead of using it as a
continuous delay -- changes gap_closed across an existing checkpoint sweep,
in particular the F1-saturation collapse region documented in
Hypotheses #16-19 (output/miniwarehouse_35a_0004_memorization).

Usage:
    python scripts/binarized_vs_continuous_checkpoints.py \
        --checkpoint-dir output/miniwarehouse_35a_0004_memorization/checkpoints \
        --dataset-dir data/miniwarehouse-35a-0004-one-scen \
        --num-seeds 8 --output-csv output/oracle_noise_sweep/binarized_vs_continuous.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from loguru import logger

from marl_path.model.inference import load_model
from marl_path.pipelines.lacam_eval import eval_test_instances, resolve_test_dir


class BinarizedModel:
    """Wraps a trained model, hard-thresholding its output at `threshold`."""

    def __init__(self, model, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return (out >= self.threshold).to(out.dtype)


class ZeroFloorModel:
    """Wraps a trained model, snapping outputs below `threshold` to 0.

    Unlike BinarizedModel, values at/above the threshold are left at their
    continuous value (asymmetric floor, not a hard 0/1 decision).
    """

    def __init__(self, model, threshold: float = 0.8):
        self.model = model
        self.threshold = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return torch.where(out < self.threshold, torch.zeros_like(out), out)


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def run_comparison(
    checkpoint_dir: Path,
    dataset_dir: Path,
    num_seeds: int,
    time_limit_ms: int,
    flg_star: bool,
    penalty_scale: float,
    seed: int,
    epoch_step: int,
    zero_floor_threshold: float = 0.8,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir, has_test = resolve_test_dir(dataset_dir)
    if not has_test:
        raise ValueError(f"No test/ data found in {dataset_dir}")

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=_checkpoint_epoch
    )
    checkpoints = checkpoints[::epoch_step]

    rows: list[dict] = []
    for ckpt_path in checkpoints:
        epoch = _checkpoint_epoch(ckpt_path)
        model, extractor = load_model(str(ckpt_path), device=device)
        model.eval()

        cont_summary = eval_test_instances(
            test_dir,
            model=model,
            device=device,
            extractor=extractor,
            time_limit_ms=time_limit_ms,
            flg_star=flg_star,
            seed=seed,
            penalty_scale=penalty_scale,
            num_seeds=num_seeds,
        )
        bin_summary = eval_test_instances(
            test_dir,
            model=BinarizedModel(model),
            device=device,
            extractor=extractor,
            time_limit_ms=time_limit_ms,
            flg_star=flg_star,
            seed=seed,
            penalty_scale=penalty_scale,
            num_seeds=num_seeds,
        )
        floor_summary = eval_test_instances(
            test_dir,
            model=ZeroFloorModel(model, threshold=zero_floor_threshold),
            device=device,
            extractor=extractor,
            time_limit_ms=time_limit_ms,
            flg_star=flg_star,
            seed=seed,
            penalty_scale=penalty_scale,
            num_seeds=num_seeds,
        )

        rows.append(
            {
                "epoch": epoch,
                "gap_closed_continuous": cont_summary.gap_closed,
                "gap_closed_binarized": bin_summary.gap_closed,
                "gap_closed_zero_floor": floor_summary.gap_closed,
                "soc_continuous": cont_summary.model.mean if cont_summary.model else None,
                "soc_binarized": bin_summary.model.mean if bin_summary.model else None,
                "soc_zero_floor": floor_summary.model.mean if floor_summary.model else None,
                "soc_baseline": cont_summary.baseline.mean,
                "soc_cbs": cont_summary.cbs_mean,
            }
        )
        logger.info(
            "epoch={:<4} gap_closed continuous={}  binarized={}  zero_floor={}",
            epoch,
            f"{100 * cont_summary.gap_closed:.1f}%" if cont_summary.gap_closed is not None else "n/a",
            f"{100 * bin_summary.gap_closed:.1f}%" if bin_summary.gap_closed is not None else "n/a",
            f"{100 * floor_summary.gap_closed:.1f}%" if floor_summary.gap_closed is not None else "n/a",
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--time-limit-ms", type=int, default=3000)
    parser.add_argument("--flg-star", action="store_true", default=False)
    parser.add_argument("--penalty-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--epoch-step",
        type=int,
        default=1,
        help="Use every Nth checkpoint (sorted by epoch) to cut runtime.",
    )
    parser.add_argument(
        "--zero-floor-threshold",
        type=float,
        default=0.8,
        help="ZeroFloorModel threshold: outputs below this are snapped to 0, "
        "outputs at/above it are left continuous.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_comparison(
        checkpoint_dir=args.checkpoint_dir,
        dataset_dir=args.dataset_dir,
        num_seeds=args.num_seeds,
        time_limit_ms=args.time_limit_ms,
        flg_star=args.flg_star,
        penalty_scale=args.penalty_scale,
        seed=args.seed,
        epoch_step=args.epoch_step,
        zero_floor_threshold=args.zero_floor_threshold,
    )
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote {}", args.output_csv)


if __name__ == "__main__":
    main()
