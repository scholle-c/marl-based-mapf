"""Feature extraction strategies for MAPF input tensor construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from marl_path.shared import Grid
from marl_path.shared.mapf_utils import get_neighbors

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
        own_start: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Return a tensor of shape (1, n_channels, H, W).

        other_starts (optional): other agents' start positions. Extractors that
        don't use this signal (BasicExtractor, BinaryAgentsChannelExtractor) may
        ignore it.
        own_start (optional): this agent's own start position (distinct from
        `start`, which is actually a duplicate of `goal` in the current calling
        convention — see CbsDataset.__getitem__). Only used by extractors that
        need to reconstruct this agent's individual shortest path, e.g. to
        detect conflicts with other agents' paths.
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
        own_start: Optional[tuple[int, int]] = None,
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
        own_start: Optional[tuple[int, int]] = None,
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
        own_start: Optional[tuple[int, int]] = None,
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
        own_start: Optional[tuple[int, int]] = None,
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


def _greedy_bfs_path(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    bfs_table: np.ndarray,
) -> list[tuple[int, int]]:
    """Reconstruct the greedy BFS-descent path from `start` to `goal`.

    At each step, move to the neighbor with the smallest distance-to-goal
    (mirrors NonAStarPenaltyDelay.compute). `bfs_table` must be anchored at
    `goal`. Guards against non-terminating loops on a disconnected/corrupt
    table with a step cap — should never trigger on a connected grid.
    """
    current = start
    path = [current]
    max_steps = grid.size + 1
    while current != goal and len(path) <= max_steps:
        neighbors = get_neighbors(grid, current)
        if not neighbors:
            break
        current = min(neighbors, key=lambda n: bfs_table[n])
        path.append(current)
    return path


def _has_vertex_conflict(
    path_a: list[tuple[int, int]], path_b: list[tuple[int, int]]
) -> bool:
    """True if both agents occupy the same cell at the same timestep.

    Shorter path is padded by holding at its last (goal) cell, matching the
    SOC-padding convention used elsewhere (e.g. _cbs_soc_from_cached_paths).
    Swap/edge conflicts are not checked (vertex conflicts only, for now).
    """
    len_diff = len(path_a) - len(path_b)
    if len_diff > 0:
        path_b = path_b + [path_b[-1]] * len_diff
    elif len_diff < 0:
        path_a = path_a + [path_a[-1]] * (-len_diff)
    return any(a == b for a, b in zip(path_a, path_b))


def _colliding_agents(
    grid: Grid,
    goal: tuple[int, int],
    own_start: Optional[tuple[int, int]],
    other_goals: list[tuple[int, int]],
    other_starts: list[tuple[int, int]],
    bfs_tables: Optional[BfsTableMap],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Filter (other_goals, other_starts) down to agents whose individual
    shortest path vertex-conflicts with the current agent's own path.

    Falls back to returning everything unfiltered when `own_start` or
    `bfs_tables` isn't available.
    """
    if own_start is None or not bfs_tables or goal not in bfs_tables:
        return other_goals, other_starts

    own_path = _greedy_bfs_path(grid, own_start, goal, bfs_tables[goal])

    colliding_goals: list[tuple[int, int]] = []
    colliding_starts: list[tuple[int, int]] = []
    for other_goal, other_start in zip(other_goals, other_starts):
        if other_goal not in bfs_tables:
            # Can't reconstruct this agent's path — include it
            # conservatively rather than silently drop its signal.
            colliding_goals.append(other_goal)
            colliding_starts.append(other_start)
            continue
        other_path = _greedy_bfs_path(
            grid, other_start, other_goal, bfs_tables[other_goal]
        )
        if _has_vertex_conflict(own_path, other_path):
            colliding_goals.append(other_goal)
            colliding_starts.append(other_start)
    return colliding_goals, colliding_starts


class CollisionAwareAgentsChannelExtractor(FeatureExtractor):
    """RichAgentsChannelExtractor, but the BFS-sum/min channels are aggregated
    only over other agents whose *individual, unconstrained* shortest path
    (start -> goal, ignoring all other agents) has a vertex conflict with this
    agent's own shortest path.

    Motivation: RichAgentsChannelExtractor squashes every other agent's
    influence into 2 scalars per pixel (BFS-sum/min), regardless of whether
    that agent's path is anywhere near this agent's path. Two very different
    multi-agent configurations can look identical to the model if their
    aggregate BFS stats happen to match. Restricting the aggregation to only
    the agents that could plausibly conflict keeps the same channel count and
    aggregation mechanism, but removes irrelevant agents from the signal.

    Falls back to RichAgentsChannelExtractor behaviour (aggregate over ALL
    other agents) when `own_start` or `bfs_tables` isn't available, or when an
    other agent's own goal BFS table is missing (can't reconstruct its path).

    Channels: identical layout to RichAgentsChannelExtractor (9 with coord
    channels, 7 without) — only the aggregation *set* differs.
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
        own_start: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        map_ch = grid.astype(np.float32, copy=False)
        goal_ch = np.zeros_like(map_ch)
        goal_ch[goal] = 1.0
        start_ch = np.zeros_like(map_ch)
        start_ch[start] = 1.0
        norm = float(grid.size) or 1.0

        colliding_goals, colliding_starts = _colliding_agents(
            grid, goal, own_start, other_agents, other_starts or [], bfs_tables
        )

        goal_sum_ch, goal_min_ch = _aggregate_bfs(
            colliding_goals, bfs_tables, map_ch, norm
        )
        start_sum_ch, start_min_ch = _aggregate_bfs(
            colliding_starts, bfs_tables, map_ch, norm
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


def _path_mask(grid: Grid, path: list[tuple[int, int]]) -> np.ndarray:
    """Binary channel: 1.0 on every cell in `path`, 0.0 elsewhere."""
    mask = np.zeros(grid.shape, dtype=np.float32)
    for coord in path:
        mask[coord] = 1.0
    return mask


class PathMembershipAgentsChannelExtractor(FeatureExtractor):
    """Encodes each agent's individual, unconstrained shortest path (start ->
    goal, greedy BFS-descent, ignoring other agents) as an explicit binary
    grid channel, instead of collapsing it into BFS-distance sum/min scalars
    like RichAgentsChannelExtractor/CollisionAwareAgentsChannelExtractor.

    Channels: map, goal, start, own_path_mask, other_paths_mask [, y_rel, x_rel]

    other_paths_mask is the union (binary OR) of other agents' path masks.
    `agents_filter` controls which other agents contribute to that union:
      "all"       — every other agent (default; avoids the sparsity problem
                    seen with collision-only filtering on open maps, where
                    most agents have zero colliding partners).
      "colliding" — only agents whose path vertex-conflicts with this agent's
                    own path (same conflict test as
                    CollisionAwareAgentsChannelExtractor).

    Falls back to an all-zero own_path_mask when `own_start`/`bfs_tables`
    aren't available (needed to reconstruct paths).
    """

    def __init__(
        self,
        agents_filter: str = "all",
        use_coord_channels: bool = True,
    ):
        if agents_filter not in ("all", "colliding"):
            raise ValueError(f"Unknown agents_filter: {agents_filter!r}")
        self._agents_filter = agents_filter
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
        own_start: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        map_ch = grid.astype(np.float32, copy=False)
        goal_ch = np.zeros_like(map_ch)
        goal_ch[goal] = 1.0
        start_ch = np.zeros_like(map_ch)
        start_ch[start] = 1.0
        other_starts = other_starts or []

        own_path_ch = np.zeros_like(map_ch)
        if own_start is not None and bfs_tables and goal in bfs_tables:
            own_path = _greedy_bfs_path(grid, own_start, goal, bfs_tables[goal])
            own_path_ch = _path_mask(grid, own_path)

        relevant_goals, relevant_starts = other_agents, other_starts
        if self._agents_filter == "colliding":
            relevant_goals, relevant_starts = _colliding_agents(
                grid, goal, own_start, other_agents, other_starts, bfs_tables
            )

        other_paths_ch = np.zeros_like(map_ch)
        if bfs_tables:
            for other_goal, other_start in zip(relevant_goals, relevant_starts):
                if other_goal not in bfs_tables:
                    continue
                other_path = _greedy_bfs_path(
                    grid, other_start, other_goal, bfs_tables[other_goal]
                )
                other_paths_ch = np.maximum(
                    other_paths_ch, _path_mask(grid, other_path)
                )

        tensor = torch.from_numpy(
            np.stack([map_ch, goal_ch, start_ch, own_path_ch, other_paths_ch])
        ).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        if self._use_coord_channels:
            tensor = _add_relative_coords(tensor, goal)
        return tensor
