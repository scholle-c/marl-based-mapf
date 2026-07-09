"""Feature extraction strategies for MAPF input tensor construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from marl_path.shared import Grid

BfsTableMap = dict[tuple[int, int], np.ndarray]


class FeatureExtractor(ABC):
    """Base class for MAPF input feature extraction strategies."""

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of channels produced by this extractor."""
        ...

    @abstractmethod
    def extract(
        self,
        grid: Grid,
        goal: tuple[int, int],
        start: tuple[int, int],
        other_agents: list[tuple[int, int]],
        device: torch.device | None = None,
        bfs_tables: Optional[BfsTableMap] = None,
        other_starts: Optional[list[tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """Return a tensor of shape (1, n_channels, H, W).

        other_starts (optional): other agents' start positions. Extractors that
        don't use this signal (BasicExtractor, BinaryAgentsChannelExtractor) may
        ignore it.
        """
        ...


def _add_relative_coords(tensor: torch.Tensor, goal: tuple[int, int]) -> torch.Tensor:
    """Append two relative coordinate channels (y, x) w.r.t. goal to (B, C, H, W)."""
    batch_size, _, height, width = tensor.shape
    target_y, target_x = goal
    y_rel = (torch.arange(height, dtype=torch.float32) - target_y) / height
    x_rel = (torch.arange(width, dtype=torch.float32) - target_x) / width
    y_coords = y_rel.view(1, 1, height, 1).expand(batch_size, 1, height, width)
    x_coords = x_rel.view(1, 1, 1, width).expand(batch_size, 1, height, width)
    y_coords = y_coords.to(tensor.device)
    x_coords = x_coords.to(tensor.device)
    return torch.cat([tensor, y_coords, x_coords], dim=1)


class BasicExtractor(FeatureExtractor):
    """Map + goal + start channels with optional relative coordinate channels.

    Reproduces the behaviour of build_input_tensor. Other agents are ignored.
    Channels: map, goal, start [, y_rel, x_rel]
    """

    def __init__(self, use_coord_channels: bool = True):
        self._use_coord_channels = use_coord_channels

    @property
    def n_channels(self) -> int:
        return 5 if self._use_coord_channels else 3

    def extract(
        self,
        grid: Grid,
        goal: tuple[int, int],
        start: tuple[int, int],
        other_agents: list[tuple[int, int]],
        device: torch.device | None = None,
        bfs_tables: Optional[BfsTableMap] = None,
        other_starts: Optional[list[tuple[int, int]]] = None,
    ) -> torch.Tensor:
        map_ch = grid.astype(np.float32, copy=False)
        goal_ch = np.zeros_like(map_ch)
        goal_ch[goal] = 1.0
        start_ch = np.zeros_like(map_ch)
        start_ch[start] = 1.0

        tensor = torch.from_numpy(np.stack([map_ch, goal_ch, start_ch])).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        if self._use_coord_channels:
            tensor = _add_relative_coords(tensor, goal)
        return tensor


class BinaryAgentsChannelExtractor(FeatureExtractor):
    """BasicExtractor plus one binary channel marking all other agent positions.

    Channels: map, goal, start, other_agents [, y_rel, x_rel]
    """

    def __init__(self, use_coord_channels: bool = True):
        self._use_coord_channels = use_coord_channels

    @property
    def n_channels(self) -> int:
        return 6 if self._use_coord_channels else 4

    def extract(
        self,
        grid: Grid,
        goal: tuple[int, int],
        start: tuple[int, int],
        other_agents: list[tuple[int, int]],
        device: torch.device | None = None,
        bfs_tables: Optional[BfsTableMap] = None,
        other_starts: Optional[list[tuple[int, int]]] = None,
    ) -> torch.Tensor:
        map_ch = grid.astype(np.float32, copy=False)
        goal_ch = np.zeros_like(map_ch)
        goal_ch[goal] = 1.0
        start_ch = np.zeros_like(map_ch)
        start_ch[start] = 1.0
        agents_ch = np.zeros_like(map_ch)
        for pos in other_agents:
            agents_ch[pos] = 1.0

        tensor = torch.from_numpy(
            np.stack([map_ch, goal_ch, start_ch, agents_ch])
        ).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        if self._use_coord_channels:
            tensor = _add_relative_coords(tensor, goal)
        return tensor


def _aggregate_bfs(
    coords: list[tuple[int, int]],
    bfs_tables: Optional[BfsTableMap],
    template: np.ndarray,
    norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum/min BFS-distance channels over `coords`' BFS tables (both normalised by `norm`).

    Sum: low value -> close to the goal/start area of many agents.
    Min: low value -> close to at least one agent (agent count is lost).
    Falls back to a binary marker channel (duplicated for sum/min) when bfs_tables is None.
    """
    if bfs_tables:
        tables = [bfs_tables[c] for c in coords if c in bfs_tables]
        if tables:
            sum_ch = (np.sum(tables, axis=0) / norm).astype(np.float32)
            min_ch = (np.minimum.reduce(tables) / norm).astype(np.float32)
        else:
            sum_ch = np.zeros_like(template)
            min_ch = np.zeros_like(template)
    else:
        sum_ch = np.zeros_like(template)
        min_ch = np.zeros_like(template)
        for pos in coords:
            sum_ch[pos] = 1.0
            min_ch[pos] = 1.0
    return sum_ch, min_ch


class AggregatedAgentsChannelExtractor(FeatureExtractor):
    """BasicExtractor plus BFS-sum and BFS-min channels over other agents' goals.

    Both channels are normalised by grid size so values stay in a comparable range.
    Channels: map, goal, start, summed_bfs, min_bfs [, y_rel, x_rel]

    Falls back to the other-agents binary channel (duplicated) when bfs_tables is None.
    """

    def __init__(self, use_coord_channels: bool = True):
        self._use_coord_channels = use_coord_channels

    @property
    def n_channels(self) -> int:
        return 7 if self._use_coord_channels else 5

    def extract(
        self,
        grid: Grid,
        goal: tuple[int, int],
        start: tuple[int, int],
        other_agents: list[tuple[int, int]],
        device: torch.device | None = None,
        bfs_tables: Optional[BfsTableMap] = None,
        other_starts: Optional[list[tuple[int, int]]] = None,
    ) -> torch.Tensor:
        map_ch = grid.astype(np.float32, copy=False)
        goal_ch = np.zeros_like(map_ch)
        goal_ch[goal] = 1.0
        start_ch = np.zeros_like(map_ch)
        start_ch[start] = 1.0
        norm = float(grid.size) or 1.0

        bfs_sum_ch, bfs_min_ch = _aggregate_bfs(other_agents, bfs_tables, map_ch, norm)

        tensor = torch.from_numpy(
            np.stack([map_ch, goal_ch, start_ch, bfs_sum_ch, bfs_min_ch])
        ).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        if self._use_coord_channels:
            tensor = _add_relative_coords(tensor, goal)
        return tensor


class RichAgentsChannelExtractor(FeatureExtractor):
    """AggregatedAgentsChannelExtractor plus BFS-sum/min channels over other agents'
    *start* positions, not just their goals.

    NonOptimalPenaltyDelay-style targets depend on which alternative paths CBS
    ruled out to avoid conflicts with other agents — that requires knowing where
    other agents *start*, not only where they're headed. Channels: map, goal,
    start, goal_bfs_sum, goal_bfs_min, start_bfs_sum, start_bfs_min [, y_rel, x_rel]
    """

    def __init__(self, use_coord_channels: bool = True):
        self._use_coord_channels = use_coord_channels

    @property
    def n_channels(self) -> int:
        return 9 if self._use_coord_channels else 7

    def extract(
        self,
        grid: Grid,
        goal: tuple[int, int],
        start: tuple[int, int],
        other_agents: list[tuple[int, int]],
        device: torch.device | None = None,
        bfs_tables: Optional[BfsTableMap] = None,
        other_starts: Optional[list[tuple[int, int]]] = None,
    ) -> torch.Tensor:
        map_ch = grid.astype(np.float32, copy=False)
        goal_ch = np.zeros_like(map_ch)
        goal_ch[goal] = 1.0
        start_ch = np.zeros_like(map_ch)
        start_ch[start] = 1.0
        norm = float(grid.size) or 1.0

        goal_sum_ch, goal_min_ch = _aggregate_bfs(
            other_agents, bfs_tables, map_ch, norm
        )
        start_sum_ch, start_min_ch = _aggregate_bfs(
            other_starts or [], bfs_tables, map_ch, norm
        )

        tensor = torch.from_numpy(
            np.stack(
                [
                    map_ch,
                    goal_ch,
                    start_ch,
                    goal_sum_ch,
                    goal_min_ch,
                    start_sum_ch,
                    start_min_ch,
                ]
            )
        ).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        if self._use_coord_channels:
            tensor = _add_relative_coords(tensor, goal)
        return tensor
