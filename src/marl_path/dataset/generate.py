"""CLI for generating an EECBS-optimal MAPF dataset.

Usage:
    marl-generate \\
        --map-file assets/random-32-32-20.map \\
        --scen-dir assets/random-32-32-20_map-scen-random/scen-random \\
        --output-dir data/random-32-32-20 \\
        --num-agents 30 --subsets-per-scen 10 --timeout 60 --suboptimality 1.2
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger

from marl_path.shared.mapf_utils import get_grid, get_scenario
from marl_path.model.training import _compute_bfs_table, _get_path_target_first_visit

from .cbs_runner import run_eecbs
from .instance import CachedInstance

_DEFAULT_EECBS = Path(__file__).parent.parent.parent.parent.parent / "EECBS" / "eecbs"


def _read_scen_data_lines(scen_file: Path) -> list[str]:
    """Return non-header lines from a scen file."""
    lines = []
    with open(scen_file) as f:
        for line in f:
            if line.startswith("version") or not line.strip():
                continue
            lines.append(line)
    return lines


def _write_subset_scen(
    data_lines: list[str], indices: list[int], out_path: Path
) -> None:
    """Write a new scen file containing only the agents at the given indices."""
    with open(out_path, "w") as f:
        f.write("version 1\n")
        for i in indices:
            f.write(data_lines[i])


def generate(
    eecbs_binary: str | Path,
    map_file: str | Path,
    scen_files: list[Path],
    output_dir: Path,
    num_agents: int = 30,
    subsets_per_scen: int = 1,
    timeout_s: float = 60.0,
    seed: int = 0,
    suboptimality: float = 1.2,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = get_grid(map_file)

    total_attempts = 0
    timeouts = 0
    total_path_cells = 0
    nonzero_delay_cells = 0

    for scen_idx, scen_file in enumerate(scen_files):
        scen_name = scen_file.stem
        data_lines = _read_scen_data_lines(scen_file)
        total_available = len(data_lines)

        if total_available < num_agents:
            logger.warning(
                f"Skipping {scen_name}: only {total_available} agents, need {num_agents}"
            )
            continue

        rng = np.random.default_rng(seed + scen_idx * 1000)

        # Build subset list: subset 0 always uses the first num_agents agents.
        # Subsequent subsets are random draws without replacement.
        subsets: list[list[int]] = [list(range(num_agents))]
        for _ in range(1, subsets_per_scen):
            indices = sorted(
                rng.choice(total_available, num_agents, replace=False).tolist()
            )
            subsets.append(indices)

        for subset_idx, agent_indices in enumerate(subsets):
            total_attempts += 1
            instance_name = f"{scen_name}_s{subset_idx:02d}"
            out_file = output_dir / f"{instance_name}.npz"

            if out_file.exists():
                logger.debug(f"  skip (exists): {instance_name}")
                continue

            logger.info(
                f"EECBS: {instance_name}  agents={agent_indices[:3]}... "
                f"(timeout={timeout_s}s, subopt={suboptimality})"
            )

            # Write a temporary scen file containing only the selected agents.
            with tempfile.NamedTemporaryFile(
                suffix=".scen", mode="w", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
            try:
                _write_subset_scen(data_lines, agent_indices, tmp_path)
                paths = run_eecbs(
                    eecbs_binary,
                    map_file,
                    tmp_path,
                    num_agents,
                    timeout_s,
                    suboptimality,
                )
            finally:
                tmp_path.unlink(missing_ok=True)

            if paths is None:
                logger.warning(f"  timeout / failure: {instance_name}")
                timeouts += 1
                continue

            # Load goals for the selected agents to compute delay stats.
            all_starts, all_goals = get_scenario(scen_file)
            goals = [all_goals[i] for i in agent_indices]

            instance = CachedInstance(
                map_file=str(map_file),
                scen_file=str(scen_file),
                agent_indices=agent_indices,
                paths=paths,
            )
            instance.save(out_file)

            # Measure delay distribution: delay(v) = h_total(v) - h_bfs(v)
            for i, path in enumerate(paths):
                bfs = _compute_bfs_table(grid, goals[i])
                targets = _get_path_target_first_visit(path)
                for coord, t in zip(path, targets):
                    total_path_cells += 1
                    if t - bfs[coord] > 1e-6:
                        nonzero_delay_cells += 1

            logger.info(f"  saved {out_file.name}")

    successes = total_attempts - timeouts
    logger.info(
        f"\nDone: {successes}/{total_attempts} instances saved ({timeouts} timeouts)"
    )
    if total_path_cells > 0:
        pct = 100.0 * nonzero_delay_cells / total_path_cells
        logger.info(
            f"Delay distribution: {nonzero_delay_cells}/{total_path_cells} "
            f"path cells have non-zero delay ({pct:.1f}%)"
        )
        if pct < 5.0:
            logger.warning(
                "Less than 5% of path cells have non-zero delay — "
                "dataset is delay-poor. Consider denser maps or clustered agent sampling."
            )
    else:
        logger.warning("No path cells recorded (all instances failed?).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an EECBS-optimal MAPF dataset for supervised delay training."
    )
    parser.add_argument(
        "--eecbs-binary",
        type=Path,
        default=_DEFAULT_EECBS if _DEFAULT_EECBS.exists() else None,
        help=f"Path to the EECBS binary. Default: {_DEFAULT_EECBS}",
    )
    parser.add_argument(
        "--suboptimality",
        type=float,
        default=1.2,
        help="Suboptimality bound for EECBS (default: 1.2).",
    )
    parser.add_argument(
        "--map-file",
        type=Path,
        required=True,
        help="Path to the .map file.",
    )
    parser.add_argument(
        "--scen-dir",
        type=Path,
        required=True,
        help="Directory containing .scen files (all *.scen files will be used).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write cached .npz instance files.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=30,
        help="Number of agents per instance (default: 30).",
    )
    parser.add_argument(
        "--subsets-per-scen",
        type=int,
        default=1,
        help=(
            "Number of agent subsets to generate per scen file. "
            "Subset 0 = first num-agents rows; subsets 1+ = random draws. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="CBS timeout per instance in seconds (default: 60).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for agent subset sampling (default: 0).",
    )

    args = parser.parse_args()

    if args.eecbs_binary is None:
        parser.error("--eecbs-binary is required (default path not found).")

    scen_files = sorted(args.scen_dir.glob("*.scen"))
    if not scen_files:
        parser.error(f"No .scen files found in {args.scen_dir}")

    expected = len(scen_files) * args.subsets_per_scen
    logger.info(
        f"Generating dataset: map={args.map_file.name}, "
        f"{len(scen_files)} scen files × {args.subsets_per_scen} subsets "
        f"= up to {expected} instances, {args.num_agents} agents each, "
        f"suboptimality={args.suboptimality}"
    )
    generate(
        eecbs_binary=args.eecbs_binary,
        map_file=args.map_file,
        scen_files=scen_files,
        output_dir=args.output_dir,
        num_agents=args.num_agents,
        subsets_per_scen=args.subsets_per_scen,
        timeout_s=args.timeout,
        seed=args.seed,
        suboptimality=args.suboptimality,
    )


if __name__ == "__main__":
    main()
