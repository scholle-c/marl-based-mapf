from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from .mapf_utils import Coord, Grid, get_neighbors, is_valid_coord
from pathfinding_model import load_model, DistanceTableCNN, build_input_tensor


@dataclass
class DistTable:
    grid: Grid
    goal: Coord
    model: DistanceTableCNN
    Q: deque = field(init=False)
    table: np.ndarray = field(init=False)  # distance matrix
    NIL: int = field(init=False)

    def __post_init__(self):
        self.NIL = self.grid.size
        self.Q = deque([self.goal])
        self.table = np.full(self.grid.shape, self.NIL, dtype=int)
        self.table[self.goal] = 0

    def get(self, start: Coord) -> int:
        # check valid input
        if not is_valid_coord(self.grid, start):
            return self.grid.size

        # distance has been known
        if self.table[start] != self.NIL:
            return self.table[start]

        # compute distance table using the model
        self.input_tensor: torch.Tensor = build_input_tensor(self.grid, self.goal, start)
        with torch.no_grad():
            output: torch.Tensor = self.model(self.input_tensor)
        dist_table: np.ndarray = output.squeeze(0).squeeze(0).cpu().numpy()
        dist_value: int = int(dist_table[start])
        self.table = dist_table.astype(int)
        return dist_value