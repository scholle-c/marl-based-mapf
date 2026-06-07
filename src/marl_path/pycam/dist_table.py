from dataclasses import dataclass, field

import numpy as np
import torch
from typing import Optional

from marl_path.shared.mapf_utils import BfsCache, Coord, Grid, is_valid_coord
from marl_path.model.definition import DefaultModel
from marl_path.model.feature_extraction import FeatureExtractor, BasicExtractor


@dataclass
class DistTable:
    grid: Grid
    goal: Coord
    model: Optional[DefaultModel] = None
    device: torch.device | None = None
    extractor: Optional[FeatureExtractor] = None
    input_tensor: Optional[torch.Tensor] = None
    other_agents: list[Coord] = field(default_factory=lambda: [])
    bfs_cache: BfsCache = field(kw_only=True, repr=False)
    table: np.ndarray = field(init=False)  # distance heuristic (BFS)
    delay: np.ndarray = field(init=False)  # delay matrix (model prediction)
    NIL: int = field(init=False)

    def __post_init__(self):
        if self.extractor is None:
            self.extractor = BasicExtractor()
        self.NIL = self.grid.size

        self.table = self.bfs_cache[self.goal].copy()

        self.delay = np.zeros(self.grid.shape, dtype=np.float32)
        if self.model is not None:
            self.delay = self.compute_delay_model(self.goal)

    def get(self, target: Coord) -> int:
        if not is_valid_coord(self.grid, target):
            return self.grid.size
        return self.table[target] + self.delay[target]

    def compute_delay_model(self, target: Coord) -> np.ndarray:
        other_bfs = {g: self.bfs_cache[g] for g in self.other_agents}
        self.input_tensor = self.extractor.extract(  # type: ignore[union-attr]
            self.grid, self.goal, target, self.other_agents, device=self.device,
            bfs_tables=other_bfs,
        )
        with torch.no_grad():
            output: torch.Tensor = self.model(self.input_tensor)  # type: ignore
        delay: np.ndarray = output.squeeze(0).squeeze(0).cpu().numpy()
        return delay  # type: ignore
