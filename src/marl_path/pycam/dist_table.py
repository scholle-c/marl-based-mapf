from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from typing import Optional

from marl_path.shared.mapf_utils import Coord, Grid, get_neighbors, is_valid_coord
from marl_path.model.definition import DistanceTableCNN
from marl_path.model.training import build_input_tensor


@dataclass
class DistTable:
    grid: Grid
    goal: Coord
    model: Optional[DistanceTableCNN] = None
    device: torch.device | None = None
    has_model_generated: bool = field(init=False, default=False)
    Q: deque = field(init=False)
    table: np.ndarray = field(init=False)  # distance matrix
    NIL: int = field(init=False)

    def __post_init__(self):
        self.NIL = self.grid.size
        self.Q = deque([self.goal])
        self.table = np.full(self.grid.shape, self.NIL, dtype=int)
        self.table[self.goal] = 0

    def get(self, target: Coord) -> int:
        # check valid input
        if not is_valid_coord(self.grid, target):
            return self.grid.size

        # distance has been known
        if self.table[target] < self.table.size or self.has_model_generated:
            return self.table[target]

        # compute distance table using either BFS or CNN model
        if self.model is None:
            return self.compute_table_bfs(target)
        else:
            self.has_model_generated = True
            return self.compute_table_model(target)  # type: ignore

    def compute_table_model(self, target: Coord) -> None:
        self.input_tensor: torch.Tensor = build_input_tensor(
            self.grid, self.goal, target, device=self.device
        )
        with torch.no_grad():
            output: torch.Tensor = self.model(self.input_tensor)  # type: ignore
        dist_table: np.ndarray = output.squeeze(0).squeeze(0).cpu().numpy()
        dist_value: int = int(dist_table[target])
        self.table = dist_table.astype(int)
        return dist_value  # type: ignore

    def compute_table_bfs(self, target: Coord) -> int:
        while len(self.Q) > 0:
            u = self.Q.popleft()
            d = int(self.table[u])
            for v in get_neighbors(self.grid, u):
                if d + 1 < self.table[v]:
                    self.table[v] = d + 1
                    self.Q.append(v)
            if u == target:
                return d
        return self.NIL
