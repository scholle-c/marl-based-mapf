"""Subprocess wrapper for EECBS solver."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from marl_path.shared import Coord


def run_eecbs(
    eecbs_binary: str | Path,
    map_file: str | Path,
    scen_file: str | Path,
    num_agents: int,
    timeout_s: float = 60.0,
    suboptimality: float = 1.2,
) -> Optional[list[list[Coord]]]:
    """Run EECBS and return per-agent paths as list[list[Coord]] in (y,x) convention.

    Returns None if EECBS times out, fails, or produces no paths file.
    EECBS outputs (row, col) = (y, x), matching this project's internal convention.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = Path(tmp) / "out.csv"
        out_paths = Path(tmp) / "paths.txt"
        cmd = [
            str(eecbs_binary),
            "-m",
            str(map_file),
            "-a",
            str(scen_file),
            "-o",
            str(out_csv),
            f"--outputPaths={out_paths}",
            "-k",
            str(num_agents),
            "-t",
            str(int(timeout_s)),
            f"--suboptimality={suboptimality}",
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout_s + 10,
            )
        except subprocess.TimeoutExpired:
            return None

        if not out_paths.exists():
            return None

        return _parse_paths(out_paths, expected_agents=num_agents)


def _parse_paths(path_file: Path, expected_agents: int) -> Optional[list[list[Coord]]]:
    agents: dict[int, list[Coord]] = {}
    with open(path_file) as f:
        for line in f:
            m = re.match(r"Agent\s+(\d+):\s*(.+)", line.strip())
            if not m:
                continue
            agent_id = int(m.group(1))
            # CBS outputs (y,x) = (row,col), same as internal convention — no swap needed.
            coords = re.findall(r"\((\d+),(\d+)\)", m.group(2))
            agents[agent_id] = [(int(y), int(x)) for y, x in coords]

    if len(agents) != expected_agents:
        return None

    return [agents[i] for i in range(expected_agents)]
