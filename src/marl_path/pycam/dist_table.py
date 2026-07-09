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
    other_agent_starts: list[Coord] = field(default_factory=lambda: [])
    penalty_scale: float = 1.0
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
        other_bfs.update({s: self.bfs_cache[s] for s in self.other_agent_starts})
        self.input_tensor = self.extractor.extract(  # type: ignore[union-attr]
            self.grid, self.goal, target, self.other_agents, device=self.device,
            bfs_tables=other_bfs, other_starts=self.other_agent_starts,
        )
        with torch.no_grad():
            output: torch.Tensor = self.model(self.input_tensor)  # type: ignore
        delay: np.ndarray = output.squeeze(0).squeeze(0).cpu().numpy()
        # Sigmoid heads output a [0,1] probability — the actual penalty magnitude
        # added to h_bfs is an explicit, independently-tunable scale (unlike the
        # softplus head, which bakes DELAY_SCALE into the activation itself).
        if getattr(self.model, "output_activation", "softplus") == "sigmoid":
            delay = delay * self.penalty_scale
        return delay  # type: ignore
