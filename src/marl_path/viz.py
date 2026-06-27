"""Interactive visualization: delay heatmap + path explorer.

Left panel:  delay heatmap (sum of all agents' delay maps).
Right panel: map grid with agent paths, switchable via radio buttons.
Title bar:   CBS / Baseline / Delay SOC values.

Primary usage — load directly from a cached dataset instance (.npz):

    marl-viz \\
        --npz-file data/random-32-32-20/my_instance.npz \\
        --delay-method first_visit

The .npz already contains the CBS paths, map reference, and agent indices,
so no EECBS binary is needed and the heatmap will be populated.

To browse a dataset directory, use --dataset-dir and optionally --instance-idx:

    marl-viz --dataset-dir data/random-32-32-20 --instance-idx 3

Fallback — provide map/scen directly (requires --eecbs-binary for CBS paths):

    marl-viz \\
        --map-file assets/random-32-32-20.map \\
        --scen-file assets/.../random-32-32-20-random-1.scen \\
        --num-agents 10 --eecbs-binary EECBS/eecbs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from loguru import logger

from marl_path.dataset.cbs_runner import run_eecbs
from marl_path.dataset.instance import CachedInstance
from marl_path.delay_methods import DELAY_METHODS, get_delay_method
from marl_path.pycam import LaCAM
from marl_path.pipeline import _get_soc
from marl_path.shared.mapf_utils import BfsCache, Config, Configs, Coord, Grid, get_grid, get_scenario


_COLORS = plt.cm.tab20.colors  # type: ignore[attr-defined]


def _color(i: int):
    return _COLORS[i % len(_COLORS)]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_from_npz(npz_path: Path) -> tuple[Grid, Config, Config, list[Coord], list[list[Coord]]]:
    """Load grid, starts, goals, and CBS paths from a cached instance file."""
    instance = CachedInstance.load(npz_path)
    grid = get_grid(instance.map_file)
    all_starts, all_goals = get_scenario(instance.scen_file)
    starts = Config(positions=[all_starts[i] for i in instance.agent_indices])
    goals = Config(positions=[all_goals[i] for i in instance.agent_indices])
    return grid, starts, goals, list(goals.positions), instance.paths


def _load_from_map_scen(
    args: argparse.Namespace,
) -> tuple[Grid, Config, Config, list[Coord], list[list[Coord]] | None, float | None]:
    """Load grid/starts/goals from map+scen files; optionally run EECBS for CBS paths."""
    grid = get_grid(args.map_file)
    starts_cfg, goals_cfg = get_scenario(args.scen_file, args.num_agents)
    starts = Config(positions=list(starts_cfg.positions))
    goals = Config(positions=list(goals_cfg.positions))
    goal_list: list[Coord] = list(goals.positions)

    cbs_paths: list[list[Coord]] | None = None
    cbs_soc: float | None = None
    eecbs = getattr(args, "eecbs_binary", None)
    if eecbs and Path(eecbs).exists():
        logger.info("Running EECBS...")
        cbs_paths = run_eecbs(eecbs, args.map_file, args.scen_file, len(goal_list), timeout_s=args.timeout)
        if cbs_paths is None:
            logger.warning("CBS timed out or failed — delay heatmap will be empty.")
        else:
            cbs_soc = float(_get_soc(_agent_paths_to_solution(cbs_paths)))
            logger.info("CBS SOC: {}", cbs_soc)
    else:
        logger.warning("No --eecbs-binary provided — delay heatmap will be empty.")

    return grid, starts, goals, goal_list, cbs_paths, cbs_soc


# ── Main computation ──────────────────────────────────────────────────────────

def compute_all(args: argparse.Namespace) -> dict:
    # ── Resolve instance source ───────────────────────────────────────────────
    npz_path: Path | None = None
    if getattr(args, "npz_file", None):
        npz_path = Path(args.npz_file)
    elif getattr(args, "dataset_dir", None):
        files = sorted(Path(args.dataset_dir).glob("*.npz"))
        if not files:
            raise ValueError(f"No .npz files found in {args.dataset_dir}")
        idx = getattr(args, "instance_idx", 0) or 0
        npz_path = files[idx]
        logger.info("Using instance {}/{}: {}", idx, len(files) - 1, npz_path.name)

    cbs_soc: float | None = None

    if npz_path is not None:
        grid, starts, goals, goal_list, cbs_paths = _load_from_npz(npz_path)
        n = len(goal_list)
        cbs_soc = float(_get_soc(_agent_paths_to_solution(cbs_paths)))
        logger.info("Loaded {} agents from {} | CBS SOC: {}", n, npz_path.name, cbs_soc)
    else:
        grid, starts, goals, goal_list, cbs_paths, cbs_soc = _load_from_map_scen(args)
        n = len(goal_list)

    # ── Delay maps ────────────────────────────────────────────────────────────
    delay_maps: list[np.ndarray] | None = None
    delay_heatmap: np.ndarray = np.where(grid, 0.0, np.nan).astype(np.float32)

    if cbs_paths is not None:
        dm = get_delay_method(args.delay_method)
        bfs = BfsCache(grid)
        delay_maps = [dm.compute(grid, bfs, cbs_paths, goal_list, i) for i in range(n)]
        delay_heatmap = np.where(grid, np.sum(delay_maps, axis=0), np.nan).astype(np.float32)

    # ── LaCAM baseline ────────────────────────────────────────────────────────
    logger.info("Running LaCAM baseline...")
    sol_b = LaCAM().solve(
        grid=grid, starts=starts, goals=goals,
        seed=args.seed, time_limit_ms=args.time_limit_ms,
        flg_star=args.flg_star, verbose=0,
    )
    baseline_paths = _solution_to_agent_paths(sol_b)
    baseline_soc = float(_get_soc(sol_b)) if sol_b else None
    logger.info("Baseline SOC: {}", baseline_soc)

    # ── LaCAM + delay ─────────────────────────────────────────────────────────
    delay_lacam_paths: list[list[Coord]] | None = None
    delay_lacam_soc: float | None = None
    if delay_maps is not None:
        logger.info("Running LaCAM + delay...")
        sol_d = LaCAM().solve(
            grid=grid, starts=starts, goals=goals,
            delay_maps=delay_maps,
            seed=args.seed, time_limit_ms=args.time_limit_ms,
            flg_star=args.flg_star, verbose=0,
        )
        delay_lacam_paths = _solution_to_agent_paths(sol_d)
        delay_lacam_soc = float(_get_soc(sol_d)) if sol_d else None
        logger.info("Delay SOC: {}", delay_lacam_soc)

    instance_label = npz_path.stem if npz_path else Path(getattr(args, "map_file", "?")).stem

    return {
        "grid": grid,
        "starts": list(starts.positions),
        "goals": goal_list,
        "cbs_paths": cbs_paths,
        "cbs_soc": cbs_soc,
        "baseline_paths": baseline_paths,
        "baseline_soc": baseline_soc,
        "delay_paths": delay_lacam_paths,
        "delay_soc": delay_lacam_soc,
        "delay_heatmap": delay_heatmap,
        "instance_label": instance_label,
    }


# ── Path conversion helpers ───────────────────────────────────────────────────

def _agent_paths_to_solution(paths: list[list[Coord]]) -> Configs:
    if not paths:
        return []
    max_t = max(len(p) for p in paths)
    padded = [p + [p[-1]] * (max_t - len(p)) for p in paths]
    return [Config(positions=[padded[a][t] for a in range(len(padded))]) for t in range(max_t)]


def _solution_to_agent_paths(solution: Configs) -> list[list[Coord]] | None:
    if not solution:
        return None
    n = len(solution[0])
    return [[solution[t][i] for t in range(len(solution))] for i in range(n)]


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_grid_bg(ax: plt.Axes, grid: Grid) -> None:
    bg = np.where(grid, 0.93, 0.15).astype(float)
    ax.imshow(bg, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_paths_on(
    ax: plt.Axes,
    grid: Grid,
    paths: list[list[Coord]] | None,
    starts: list[Coord],
    goals: list[Coord],
    title: str,
) -> None:
    _draw_grid_bg(ax, grid)
    ax.set_title(title, fontsize=10)
    if paths is None:
        ax.text(0.5, 0.5, "No solution", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="red")
        return
    for i, path in enumerate(paths):
        c = _color(i)
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, color=c, linewidth=1.5, alpha=0.75)
        ax.plot(starts[i][1], starts[i][0], "o", color=c, markersize=6, zorder=4)
        ax.plot(goals[i][1],  goals[i][0],  "*", color=c, markersize=9, zorder=4)


# ── Figure ────────────────────────────────────────────────────────────────────

def _soc_label(val: float | None, name: str) -> str:
    return f"{name}: {int(val)}" if val is not None else f"{name}: FAIL"


def show(args: argparse.Namespace, data: dict) -> None:
    grid: Grid = data["grid"]

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"{data['instance_label']}  |  {len(data['goals'])} agents"
        f"  |  delay={args.delay_method}  |  seed={args.seed}  |  "
        + _soc_label(data["cbs_soc"], "CBS")
        + "   " + _soc_label(data["baseline_soc"], "Baseline")
        + "   " + _soc_label(data["delay_soc"], "Delay"),
        fontsize=10,
    )

    ax_heat  = fig.add_axes([0.04, 0.20, 0.41, 0.72])
    ax_paths = fig.add_axes([0.54, 0.20, 0.41, 0.72])
    ax_radio = fig.add_axes([0.25, 0.02, 0.50, 0.12])

    # Left: heatmap
    cmap = plt.cm.YlOrRd.copy()  # type: ignore[attr-defined]
    cmap.set_bad(color=(0.15, 0.15, 0.15))
    masked = np.ma.masked_invalid(data["delay_heatmap"])
    im = ax_heat.imshow(masked, cmap=cmap, origin="upper", interpolation="nearest")
    ax_heat.set_title("Delay heatmap (sum over agents)", fontsize=10)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # Right: path panel
    path_options: dict[str, list[list[Coord]] | None] = {
        "CBS":      data["cbs_paths"],
        "Baseline": data["baseline_paths"],
        "Delay":    data["delay_paths"],
    }
    labels = list(path_options.keys())
    active_idx = next((i for i, k in enumerate(labels) if path_options[k] is not None), 1)
    _draw_paths_on(ax_paths, grid, path_options[labels[active_idx]],
                   data["starts"], data["goals"], f"{labels[active_idx]} paths")

    # Radio buttons
    radio = RadioButtons(ax_radio, labels, active=active_idx)

    def on_select(label: str) -> None:
        ax_paths.cla()
        _draw_paths_on(ax_paths, grid, path_options[label],
                       data["starts"], data["goals"], f"{label} paths")
        fig.canvas.draw_idle()

    radio.on_clicked(on_select)
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive delay heatmap + path explorer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Instance source (pick one):\n"
            "  --npz-file PATH          single cached instance (recommended)\n"
            "  --dataset-dir DIR        pick from a dataset folder\n"
            "  --map-file + --scen-file raw map/scen (needs --eecbs-binary for heatmap)\n"
        ),
    )

    # Instance source
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--npz-file",     type=Path, help="Path to a single .npz cached instance.")
    src.add_argument("--dataset-dir",  type=Path, help="Dataset directory; use with --instance-idx.")
    parser.add_argument("--instance-idx", type=int, default=0,
                        help="Index of the instance in --dataset-dir (default: 0).")
    # Fallback: raw map/scen
    parser.add_argument("--map-file",      type=Path)
    parser.add_argument("--scen-file",     type=Path)
    parser.add_argument("-N", "--num-agents", type=int, default=10)
    parser.add_argument("--eecbs-binary",  type=Path, default=None,
                        help="EECBS binary (only needed with --map-file).")
    parser.add_argument("--timeout",       type=float, default=60.0)

    # Delay + solver
    parser.add_argument("--delay-method", type=str, default="first_visit",
                        choices=list(DELAY_METHODS))
    parser.add_argument("-s", "--seed",          type=int,  default=0)
    parser.add_argument("-t", "--time-limit-ms", type=int,  default=5000)
    parser.add_argument("--flg-star", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    # Validate: at least one source must be given
    if not args.npz_file and not args.dataset_dir and not (args.map_file and args.scen_file):
        parser.error("Provide --npz-file, --dataset-dir, or --map-file + --scen-file.")

    logger.info("Computing — this may take a moment...")
    data = compute_all(args)
    show(args, data)


if __name__ == "__main__":
    main()
