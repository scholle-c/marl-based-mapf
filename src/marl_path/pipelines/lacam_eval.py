"""LaCAM SOC evaluation: trained model vs. vanilla baseline vs. CBS-optimal.

Shared between SupervisedDelayPipeline's embedded Track B (run periodically
during training) and EvalOnlyPipeline (standalone evaluation of an already
trained --model-file, decoupled from any training loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from marl_path.pycam import LaCAM
from marl_path.model import get_soc
from marl_path.dataset.instance import CachedInstance
from marl_path.shared.mapf_utils import Config, get_grid, get_scenario


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


def resolve_test_dir(dataset_dir: str | Path) -> tuple[Path, bool]:
    """Resolve dataset_dir/test and whether it has any cached instances."""
    test_dir = Path(dataset_dir) / "test"
    has_test_data = test_dir.exists() and any(test_dir.glob("*.npz"))
    return test_dir, has_test_data


def track_b_log_suffix(eval_summary: LacamComparisonSummary) -> str:
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


def _cbs_soc_from_cached_paths(paths: list[list]) -> float:
    """Compute SOC from CBS paths stored in agent-first format (paths[agent][t])."""
    if not paths:
        return 0.0
    max_len = max(len(p) for p in paths)
    padded = [p + [p[-1]] * (max_len - len(p)) for p in paths]
    solution = [[padded[a][t] for a in range(len(padded))] for t in range(max_len)]
    return float(get_soc(solution))


def eval_test_instances(
    test_dir: Path,
    model=None,
    device=None,
    extractor=None,
    time_limit_ms: int = 3000,
    flg_star: bool = True,
    seed: int = 0,
    penalty_scale: float = 1.0,
    limit: int | None = None,
) -> LacamComparisonSummary:
    """Evaluate LaCAM (baseline vs model) on cached test instances.

    Each test instance is run once with the given seed. CBS-optimal SOC is
    read directly from the cached paths — no CBS re-execution needed.

    `limit`: if set, only the first `limit` instances (sorted by filename)
    are evaluated instead of the full test set — useful for a quick check
    before committing to a full (potentially slow) run over everything.
    """
    baseline_socs: list[float] = []
    model_socs: list[float] = []
    cbs_socs: list[float] = []
    n_instances = 0

    npz_paths = sorted(test_dir.glob("*.npz"))
    if limit is not None:
        npz_paths = npz_paths[:limit]
    

    for npz_path in npz_paths:
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
        
        logger.info(
            "Test instance {}: soc_baseline={}  soc_model={}  soc_cbs={}",
            npz_path.name,
            b_soc if b_soc is not None else "FAIL",
            m_soc if m_soc is not None else "FAIL",
            cbs_soc,
        )

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
