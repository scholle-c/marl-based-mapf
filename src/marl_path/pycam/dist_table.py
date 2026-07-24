from dataclasses import dataclass, field

import numpy as np
import torch
from typing import Optional

from marl_path.shared.mapf_utils import BfsCache, Coord, Grid, is_valid_coord
from marl_path.model.definition import DefaultModel
from marl_path.model.feature_extraction import FeatureExtractor, BasicExtractor, FovPathExtractor


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
        # Own goal's BFS table, needed by extractors that reconstruct this
        # agent's individual shortest path (own_start -> goal), e.g.
        # PathMembershipAgentsChannelExtractor/FovPathExtractor. Without this,
        # `goal in bfs_tables` is False and those extractors silently fall
        # back to an all-zero own-path signal — a train/inference mismatch,
        # since CbsDataset always supplies it during training.
        other_bfs[self.goal] = self.bfs_cache[self.goal]

        if isinstance(self.extractor, FovPathExtractor):
            tokens = self.extractor.extract_tokens(
                self.grid, self.goal, self.other_agents,
                other_starts=self.other_agent_starts, own_start=self.own_start,
                bfs_tables=other_bfs, device=self.device,
            )
            delay = np.zeros(self.grid.shape, dtype=np.float32)
            if tokens.coords.shape[0] > 0:
                with torch.no_grad():
                    probs: np.ndarray = (
                        self.model(  # type: ignore
                            tokens.features.unsqueeze(0), tokens.coords.unsqueeze(0)
                        )
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                ys = tokens.coords[:, 0].cpu().numpy()
                xs = tokens.coords[:, 1].cpu().numpy()
                # Cells outside the FOV keep delay=0 (unchanged BFS heuristic) by
                # construction — the model only ever adjusts cells it was shown.
                delay[ys, xs] = probs
            return delay * self.penalty_scale

        self.input_tensor = self.extractor.extract(  # type: ignore[union-attr]
            self.grid, self.goal, target, self.other_agents, device=self.device,
            bfs_tables=other_bfs, other_starts=self.other_agent_starts,
            own_start=self.own_start,
        )
        with torch.no_grad():
            output: torch.Tensor = self.model(self.input_tensor)  # type: ignore
        delay: np.ndarray = output.squeeze(0).squeeze(0).cpu().numpy()
        # Model outputs a [0,1] probability — the actual penalty magnitude added
        # to h_bfs is an explicit, independently-tunable scale.
        delay = delay * self.penalty_scale
        return delay  # type: ignore
