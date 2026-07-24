"""Oracle-heuristic noise sweep on larger scenarios (throwaway experiment script).

Extends the 2026-07-23 noise sweep (done on mini-warehouse instance 0004
alone, script not committed there either) to check whether independent
per-cell noise on the *exact* CBS-derived delay target still causes a large
drop in gap_closed once agent count / map size go up (mini-warehouse
35 agents, empty-32-32 70 agents) — or whether that was an artifact of the
small single-instance test.

Instead of a trained model, DistTable's `model` slot is fed an
OracleDelayModel that returns the exact delay_method target (same method/
penalty_scale used for training on that dataset, see used_config.json) plus
independent per-cell noise U(0, magnitude), bypassing any learned model.

Usage:
    python scripts/oracle_noise_sweep.py \
        --dataset-dir data/miniwarehouse-35a-0004-one-scen \
        --magnitudes 0,0.01,0.03,0.05,0.10 --num-seeds 8

    python scripts/oracle_noise_sweep.py \
        --dataset-dir data/empty32-70agents-50sub \
        --magnitudes 0,0.01,0.03,0.05,0.10 --num-seeds 3 --instance-limit 10
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from marl_path.delay_methods import get_delay_method
from marl_path.dataset.instance import CachedInstance
from marl_path.model import get_soc
from marl_path.pipelines.lacam_eval import _cbs_soc_from_cached_paths, resolve_test_dir
from marl_path.pycam import LaCAM
from marl_path.shared.mapf_utils import BfsCache, Config, get_grid, get_scenario


class OracleDelayModel:
    """Fake 'model' returning precomputed (optionally noised) delay maps.

    DistTable builds one of these per agent inside `enumerate(goals)`, in
    order, and calls `model(input_tensor)` exactly once each — so popping a
    plain queue in that same order reproduces the right map per agent
    without needing to inspect the (here unused) input tensor.
    """

    def __init__(self, delay_maps: list[np.ndarray]):
        self._queue: deque[np.ndarray] = deque(delay_maps)

    def __call__(self, _input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self._queue.popleft()).unsqueeze(0).unsqueeze(0)


def _load_instance(npz_path: Path):
    instance = CachedInstance.load(npz_path)
    grid = get_grid(instance.map_file)
    all_starts, all_goals = get_scenario(instance.scen_file)
    starts = Config(positions=[all_starts[i] for i in instance.agent_indices])
    goals_cfg = Config(positions=[all_goals[i] for i in instance.agent_indices])
    return instance, grid, starts, goals_cfg, list(goals_cfg.positions)


def _exact_delay_maps(grid, bfs_cache, paths, goals, delay_method) -> list[np.ndarray]:
    return [
        delay_method.compute(grid, bfs_cache, paths, goals, i).astype(np.float32)
        for i in range(len(paths))
    ]


def _soc(configs) -> float | None:
    return float(get_soc(configs)) if configs else None


def run_sweep(
    dataset_dir: Path,
    delay_method_name: str,
    magnitudes: list[float],
    num_seeds: int,
    time_limit_ms: int,
    flg_star: bool,
    seed: int,
    instance_limit: int | None,
    round_to_int: bool = False,
    zero_floor: float | None = None,
) -> list[dict]:
    test_dir, has_test = resolve_test_dir(dataset_dir)
    if not has_test:
        raise ValueError(f"No test/ data found in {dataset_dir}")
    npz_paths = sorted(test_dir.glob("*.npz"))
    if instance_limit is not None:
        npz_paths = npz_paths[:instance_limit]
    delay_method = get_delay_method(delay_method_name)

    # Baseline (vanilla LaCAM) and CBS-optimal don't depend on noise
    # magnitude — compute once per (instance, seed) and reuse across the
    # magnitude sweep, along with each instance's exact (unnoised) delay maps.
    cached: list[dict] = []
    for npz_path in npz_paths:
        instance, grid, starts, goals_cfg, goals = _load_instance(npz_path)
        cbs_soc = _cbs_soc_from_cached_paths(instance.paths)
        exact_maps = _exact_delay_maps(
            grid, BfsCache(grid), instance.paths, goals, delay_method
        )
        baseline_socs = []
        for seed_offset in range(num_seeds):
            run_seed = seed + seed_offset
            baseline = LaCAM().solve(
                grid=grid,
                starts=starts,
                goals=goals_cfg,
                seed=run_seed,
                time_limit_ms=time_limit_ms,
                flg_star=flg_star,
                verbose=0,
            )
            baseline_socs.append(_soc(baseline))
        cached.append(
            {
                "grid": grid,
                "starts": starts,
                "goals_cfg": goals_cfg,
                "exact_maps": exact_maps,
                "cbs_soc": cbs_soc,
                "baseline_socs": baseline_socs,
                "name": npz_path.name,
            }
        )
        logger.info(
            "Loaded {} ({} agents): cbs_soc={:.1f}  baseline_soc={:.1f}",
            npz_path.name,
            len(goals),
            cbs_soc,
            float(np.mean([s for s in baseline_socs if s is not None])),
        )

    rows: list[dict] = []
    for mag_idx, magnitude in enumerate(magnitudes):
        rng = np.random.default_rng(seed + 1000 * (mag_idx + 1))
        oracle_socs_all: list[float] = []
        baseline_socs_all: list[float] = []
        cbs_socs_all: list[float] = []

        for entry in cached:
            for seed_offset in range(num_seeds):
                run_seed = seed + seed_offset
                if magnitude > 0:
                    noised = [
                        m + rng.uniform(0.0, magnitude, size=m.shape).astype(np.float32)
                        for m in entry["exact_maps"]
                    ]
                else:
                    noised = entry["exact_maps"]
                if round_to_int:
                    noised = [np.round(m).astype(np.float32) for m in noised]
                if zero_floor is not None:
                    noised = [
                        np.where(m < zero_floor, 0.0, m).astype(np.float32)
                        for m in noised
                    ]
                oracle_solution = LaCAM().solve(
                    grid=entry["grid"],
                    starts=entry["starts"],
                    goals=entry["goals_cfg"],
                    model=OracleDelayModel(noised),
                    seed=run_seed,
                    time_limit_ms=time_limit_ms,
                    flg_star=flg_star,
                    verbose=0,
                    penalty_scale=1.0,
                )
                o = _soc(oracle_solution)
                b = entry["baseline_socs"][seed_offset]
                if o is not None:
                    oracle_socs_all.append(o)
                if b is not None:
                    baseline_socs_all.append(b)
            cbs_socs_all.append(entry["cbs_soc"])

        baseline_mean = float(np.mean(baseline_socs_all)) if baseline_socs_all else float("nan")
        oracle_mean = float(np.mean(oracle_socs_all)) if oracle_socs_all else float("nan")
        cbs_mean = float(np.mean(cbs_socs_all)) if cbs_socs_all else float("nan")
        denom = baseline_mean - cbs_mean
        gap_closed = (
            (baseline_mean - oracle_mean) / denom
            if np.isfinite(denom) and abs(denom) > 1e-9
            else None
        )
        rows.append(
            {
                "magnitude": magnitude,
                "n_instances": len(cached),
                "n_runs": len(oracle_socs_all),
                "baseline_soc_mean": baseline_mean,
                "oracle_soc_mean": oracle_mean,
                "cbs_soc_mean": cbs_mean,
                "gap_closed": gap_closed,
            }
        )
        logger.info(
            "magnitude={:<6}  baseline={:.1f}  oracle={:.1f}  cbs={:.1f}  gap_closed={}",
            magnitude,
            baseline_mean,
            oracle_mean,
            cbs_mean,
            f"{100 * gap_closed:.1f}%" if gap_closed is not None else "n/a",
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--delay-method", type=str, default="non_optimal_penalty")
    parser.add_argument(
        "--magnitudes",
        type=str,
        default="0,0.01,0.03,0.05,0.10",
        help="Comma-separated noise magnitudes m for U(0, m) added per cell.",
    )
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--time-limit-ms", type=int, default=3000)
    parser.add_argument("--flg-star", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=None,
        help="Cap number of test instances used (all test/*.npz if omitted).",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--round",
        action="store_true",
        default=False,
        help="Round noised delay maps to the nearest integer before use "
        "(sanity check: for a binary target and magnitude < 0.5 this exactly "
        "undoes the added noise).",
    )
    parser.add_argument(
        "--zero-floor",
        type=float,
        default=None,
        help="Snap noised values below this threshold to 0; values at/above "
        "it are left untouched (asymmetric floor, unlike --round which also "
        "rounds large values up to 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    magnitudes = [float(m) for m in args.magnitudes.split(",")]
    rows = run_sweep(
        dataset_dir=args.dataset_dir,
        delay_method_name=args.delay_method,
        magnitudes=magnitudes,
        num_seeds=args.num_seeds,
        time_limit_ms=args.time_limit_ms,
        flg_star=args.flg_star,
        seed=args.seed,
        instance_limit=args.instance_limit,
        round_to_int=args.round,
        zero_floor=args.zero_floor,
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
