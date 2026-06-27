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
    table: np.ndarray = field(init=False)
    NIL: int = field(init=False)

    def __post_init__(self):
        self.NIL = self.grid.size
        self.table = self.bfs_cache[self.goal].copy()
        if self.delay_map is None:
            self.delay_map = np.zeros(self.grid.shape, dtype=np.float32)

    def get(self, target: Coord) -> int:
        if not is_valid_coord(self.grid, target):
            return self.NIL
        return self.table[target] + self.delay_map[target]
