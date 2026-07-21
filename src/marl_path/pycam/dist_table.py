from dataclasses import dataclass, field

import numpy as np
import torch
from typing import Optional

from marl_path.shared.mapf_utils import BfsCache, Coord, Grid, is_valid_coord, greedy_bfs_path
from marl_path.model.definition import DefaultModel
from marl_path.model.feature_extraction import FeatureExtractor, BasicExtractor
from marl_path.delay_methods import get_delay_method


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
    own_start: Optional[Coord] = None
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
        other_bfs[self.goal] = self.bfs_cache[self.goal]
        self.input_tensor = self.extractor.extract(  # type: ignore[union-attr]
            self.grid, self.goal, target, self.other_agents, device=self.device,
            bfs_tables=other_bfs, other_starts=self.other_agent_starts,
            own_start=self.own_start,
        )
        with torch.no_grad():
            output: torch.Tensor = self.model(self.input_tensor)  # type: ignore
        pred: np.ndarray = output.squeeze(0).squeeze(0).cpu().numpy()

        if self._is_delta_target() and self.own_start is not None:
            # Model predicts a *correction* relative to the agent's own
            # unconstrained greedy path, not the on/off-path value directly
            # (see NonOptimalPenaltyDeltaDelay) — XOR-recombine with that
            # greedy path to reconstruct an actual [0,1] delay value.
            greedy_path = greedy_bfs_path(
                self.grid, self.own_start, self.goal, other_bfs[self.goal]
            )
            base = np.ones(self.grid.shape, dtype=np.float32)
            for coord in greedy_path:
                base[coord] = 0.0
            pred = base + pred - 2.0 * base * pred

        # Model outputs a [0,1] probability — the actual penalty magnitude added
        # to h_bfs is an explicit, independently-tunable scale.
        delay: np.ndarray = pred * self.penalty_scale
        return delay  # type: ignore

    def _is_delta_target(self) -> bool:
        name = getattr(self.model, "_delay_target", None)
        if not name:
            return False
        try:
            return get_delay_method(name).is_delta_target
        except ValueError:
            return False
