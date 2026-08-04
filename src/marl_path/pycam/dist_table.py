from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from marl_path.shared.mapf_utils import BfsCache, Coord, Grid, is_valid_coord


@dataclass
class DistTable:
    grid: Grid
    goal: Coord
    bfs_cache: BfsCache = field(kw_only=True, repr=False)
    delay_map: Optional[np.ndarray] = None
    delay_method: str = "max"
    cbs_path_penalty: float = 100000.0
    table: np.ndarray = field(init=False)
    NIL: int = field(init=False)

    def __post_init__(self):
        self.NIL = self.grid.size
        self.table = self.bfs_cache[self.goal].copy()
        if self.delay_map is None:
            self.delay_map = np.zeros(self.grid.shape, dtype=np.float32)

    def get(self, target: Coord, timestep: int | None = None) -> int:
        if not is_valid_coord(self.grid, target):
            return self.NIL

        assert self.delay_map is not None

        if self.delay_method != "cbs_path":
            return self.table[target] + self.delay_map[target]

        if timestep is None:
            raise ValueError("timestep must be provided for cbs_path delay method")

        # Target equals the cbs path coordinate at this timestep --> no penalty,
        # otherwise add `cbs_path_penalty`. A large penalty makes this a hard
        # constraint, a small one a hint — the axis swept in the noise matrix.
        if timestep < len(self.delay_map) and tuple(self.delay_map[timestep]) == target:
            return self.table[target] + 0
        else:
            return self.table[target] + self.cbs_path_penalty
