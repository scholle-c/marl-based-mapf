"""Evaluation pipeline: compare LaCAM with vs. without a precomputed delay table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from marl_path.dataset.instance import CachedInstance
from marl_path.delay_methods import DelayMethod
from marl_path.path_noise import (
    apply_path_noise,
    displacement_stats,
    divergence_stats,
)
from marl_path.pycam import LaCAM
from marl_path.shared.mapf_utils import (
    BfsCache,
    Config,
    Configs,
    get_grid,
    get_scenario,
)


@dataclass
class InstanceResult:
    name: str
    soc_baseline: float | None
    soc_delay: float | None
    soc_cbs: float | None

    @property
    def gap_closed(self) -> float | None:
        if None in (self.soc_baseline, self.soc_delay, self.soc_cbs):
            return None
        denom = self.soc_baseline - self.soc_cbs  # type: ignore[operator]
        if abs(denom) < 1e-9:
            return None
        return (self.soc_baseline - self.soc_delay) / denom  # type: ignore[operator]


def run_evaluation(
    args: argparse.Namespace, delay_method: DelayMethod
) -> list[InstanceResult]:
    dataset_dir = Path(args.dataset_dir)
    npz_files = sorted(dataset_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files found in {dataset_dir}")

    logger.info(
        "Evaluating {} instances from {}, delay_method={}, seed={}, time_limit={}ms",
        len(npz_files),
        dataset_dir,
        type(delay_method).__name__,
        args.seed,
        args.time_limit_ms,
    )

    results: list[InstanceResult] = []
    delay_method_name = (
        args.delay_method if args.delay_method else type(delay_method).__name__
    )

    for npz_path in npz_files:
        instance = CachedInstance.load(npz_path)
        grid = get_grid(instance.map_file)
        all_starts, all_scen_goals = get_scenario(instance.scen_file)

        starts = Config(positions=[all_starts[i] for i in instance.agent_indices])
        goals = Config(positions=[all_scen_goals[i] for i in instance.agent_indices])
        all_goals = [all_scen_goals[i] for i in instance.agent_indices]

        bfs_cache = BfsCache(grid)

        # Perturbed copy for the heuristic only — soc_cbs below must stay based
        # on the true CBS paths, otherwise the reference moves with the noise.
        heuristic_paths = apply_path_noise(
            instance.paths,
            noise=getattr(args, "path_noise", "none"),
            level=getattr(args, "path_noise_level", 0.0),
            ops=getattr(args, "path_noise_ops", 1),
            seed=getattr(args, "path_noise_seed", 0),
            p=getattr(args, "path_noise_p", 0.05),
            grid=grid,
            bfs_cache=bfs_cache,
            goals=all_goals,
        )

        # Realized error magnitude — the nominal op count / error rate are not
        # monotone, so the sweep aggregates on these instead. `shift_*` only
        # applies to route-preserving noise, `div_*` to any operator.
        shift_mean, shift_max = displacement_stats(instance.paths, heuristic_paths)
        div_frac, div_len = divergence_stats(instance.paths, heuristic_paths)
        logger.info(
            "{}: noise_shift mean={:.4f} max={} div_frac={:.4f} div_len={:.4f}",
            npz_path.stem,
            shift_mean,
            shift_max,
            div_frac,
            div_len,
        )

        delay_maps = [
            delay_method.compute(grid, bfs_cache, heuristic_paths, all_goals, agent_idx)
            for agent_idx in range(len(heuristic_paths))
        ]

        # Rolling horizon: keep only the first H timesteps of the CBS path. Past
        # that, DistTable.get adds the penalty to *every* candidate, and since
        # PIBT only sorts the neighbours of one agent at one timestep (pibt.py),
        # a uniform offset is a no-op — guidance degrades to plain BFS rather
        # than to garbage. Answers "how far ahead does the heuristic have to
        # see?" without training anything.
        horizon = getattr(args, "cbs_path_horizon", 0)
        if delay_method_name == "cbs_path" and horizon > 0:
            delay_maps = [dm[:horizon] for dm in delay_maps]

        # A horizon of 10 means something very different on a 12x12 map than on
        # Paris, so log the path lengths H should be read relative to.
        _lens = [len(p) - 1 for p in instance.paths if p]
        logger.info(
            "{}: path_len mean={:.2f} max={}",
            npz_path.stem,
            sum(_lens) / len(_lens) if _lens else 0.0,
            max(_lens, default=0),
        )

        soc_baseline = _run_lacam(
            grid, starts, goals, args.seed, args.time_limit_ms, args.flg_star
        )
        soc_delay = _run_lacam(
            grid,
            starts,
            goals,
            args.seed,
            args.time_limit_ms,
            args.flg_star,
            delay_maps=delay_maps,
            delay_method=delay_method_name,
            cbs_path_penalty=getattr(args, "cbs_path_penalty", 100000.0),
        )
        soc_cbs = _soc_from_paths(instance.paths)

        result = InstanceResult(
            name=npz_path.stem,
            soc_baseline=soc_baseline,
            soc_delay=soc_delay,
            soc_cbs=soc_cbs,
        )
        results.append(result)

        gap_str = (
            f"  gap_closed={100 * result.gap_closed:.1f}%"
            if result.gap_closed is not None
            else ""
        )
        logger.info(
            "{}: baseline={} delay={} cbs={}{}",
            npz_path.stem,
            f"{soc_baseline:.0f}" if soc_baseline is not None else "FAIL",
            f"{soc_delay:.0f}" if soc_delay is not None else "FAIL",
            f"{soc_cbs:.0f}",
            gap_str,
        )

    _log_summary(results)
    return results


def _run_lacam(
    grid,
    starts: Config,
    goals: Config,
    seed: int,
    time_limit_ms: int,
    flg_star: bool,
    delay_maps: list[np.ndarray] | None = None,
    delay_method: str = "max",
    cbs_path_penalty: float = 100000.0,
) -> float | None:
    planner = LaCAM()
    solution = planner.solve(
        grid=grid,
        starts=starts,
        goals=goals,
        delay_maps=delay_maps,
        delay_method=delay_method,
        cbs_path_penalty=cbs_path_penalty,
        seed=seed,
        time_limit_ms=time_limit_ms,
        flg_star=flg_star,
        verbose=0,
    )
    return float(_get_soc(solution)) if solution else None


def _soc_from_paths(paths: list[list]) -> float:
    """Compute SOC from agent-first CBS paths."""
    if not paths:
        return 0.0
    max_len = max(len(p) for p in paths)
    padded = [p + [p[-1]] * (max_len - len(p)) for p in paths]
    solution: Configs = [
        [padded[a][t] for a in range(len(padded))] for t in range(max_len)
    ]  # type: ignore[misc]
    return float(_get_soc(solution))


def _get_soc(solution: Configs) -> int:
    if not solution:
        return 0
    soc = 0
    num_agents = len(solution[0])
    for agent_idx in range(num_agents):
        goal = solution[-1][agent_idx]
        for t in range(1, len(solution)):
            if not (solution[t][agent_idx] == solution[t - 1][agent_idx] == goal):
                soc += 1
    return soc


def _log_summary(results: list[InstanceResult]) -> None:
    valid = [
        r for r in results if r.soc_baseline is not None and r.soc_delay is not None
    ]
    if not valid:
        logger.warning("No successful runs to summarize.")
        return

    baselines: list[float] = [r.soc_baseline for r in valid]  # type: ignore[misc]
    delays: list[float] = [r.soc_delay for r in valid]  # type: ignore[misc]
    cbss: list[float] = [r.soc_cbs for r in valid if r.soc_cbs is not None]  # type: ignore[misc]
    gaps: list[float] = [r.gap_closed for r in valid if r.gap_closed is not None]  # type: ignore[misc]

    logger.info(
        "\n=== Summary ({}/{} instances solved) ===\n"
        "  baseline : mean={:.1f}  std={:.1f},  median={:.1f}\n"
        "  delay    : mean={:.1f}  std={:.1f},  median={:.1f}\n"
        "  cbs      : mean={:.1f}, \n"
        "  gap_closed: mean={:.1f}%  (win_rate={:.1f}%)",
        len(valid),
        len(results),
        float(np.mean(baselines)),
        float(np.std(baselines)),
        float(np.median(baselines)),
        float(np.mean(delays)),
        float(np.std(delays)),
        float(np.median(delays)),
        float(np.mean(cbss)) if cbss else float("nan"),
        float(np.mean(gaps) * 100) if gaps else float("nan"),
        float(np.mean([d < b for d, b in zip(delays, baselines)]) * 100),
    )
