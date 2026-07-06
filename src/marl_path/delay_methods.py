"""Pluggable delay-map computation methods.

Each method takes the CBS paths for all agents and produces a per-cell
delay map (same shape as the grid) for one agent. The delay is added on
top of the BFS distance inside DistTable:

    h_total(v) = h_bfs(v) + delay(v)

To add a new method:
1. Subclass DelayMethod and implement compute().
2. Register it in DELAY_METHODS at the bottom of this file.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod

import numpy as np

from marl_path.shared.mapf_utils import BfsCache, Coord, Grid, get_neighbors


class DelayMethod(ABC):
    @abstractmethod
    def compute(
        self,
        grid: Grid,
        bfs_cache: BfsCache,
        paths: list[list[Coord]],
        goals: list[Coord],
        agent_idx: int,
    ) -> np.ndarray:
        """Return a per-cell delay map (shape = grid.shape) for agent_idx."""
        ...


class ZeroDelay(DelayMethod):
    """No delay — equivalent to plain BFS heuristic (baseline)."""

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        return np.zeros(grid.shape, dtype=np.float32)


class FirstVisitDelay(DelayMethod):
    """Delay = CBS cost-to-go minus BFS distance, at first-visit path cells.

    For each cell v visited by the agent's CBS path:
        delay(v) = (path_len - 1 - t_first_visit(v)) - bfs(v)

    Cells not on the path keep delay = 0. Negative delays are clamped to 0
    (the CBS path can never be faster than BFS).
    """

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        path = paths[agent_idx]
        bfs_table = bfs_cache[goals[agent_idx]]
        delay_map = np.zeros(grid.shape, dtype=np.float32)
        if not path:
            return delay_map

        total = len(path) - 1
        first_visit: dict[Coord, int] = {}
        for t, coord in enumerate(path):
            if coord not in first_visit:
                first_visit[coord] = t

        for coord, t in first_visit.items():
            cbs_cost_to_go = total - t
            delay_map[coord] = max(0.0, float(cbs_cost_to_go - bfs_table[coord]))

        return delay_map


class NonOptimalPenaltyDelay(DelayMethod):
    """Delay = Penalty (e.g. +1) for each cell that isnt on the optimal path of the agent.

    I want to see if this can even improve LaCAM's performance generally for all maps (this would prove that a general delay exist for improvement).
    The idea is that if the agent is not on the optimal path, it should be penalized to encourage it to get back on track.
    """

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        path = paths[agent_idx]
        delay_map = np.ones(grid.shape, dtype=np.float32)
        if not path:
            return delay_map

        for coord in path:
            delay_map[coord] = 0.0  # No penalty for cells on the optimal path

        return delay_map


class NonAStarPenaltyDelay(DelayMethod):
    """Delay = Penalty (e.g. +1) for each cell that isnt on the A* path of the agent.

    Is used to compare it with the NonOptimalPenaltyDelay to see how important it is to have the optimal path from CBS as a reference for the delay.
    """

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        delay_map = np.ones(grid.shape, dtype=np.float32)
        if not paths[agent_idx]:
            return delay_map

        bfs_table = bfs_cache[goals[agent_idx]]
        goal = goals[agent_idx]

        # Reconstruct the greedy BFS-descent path from start to goal: at each
        # step, move to the neighbor with the smallest distance-to-goal.
        current = paths[agent_idx][0]
        a_star_path = [current]
        while current != goal:
            neighbors = get_neighbors(grid, current)
            current = min(neighbors, key=lambda n: bfs_table[n])
            a_star_path.append(current)

        for coord in a_star_path:
            delay_map[coord] = 0.0  # No penalty for cells on the A* path

        return delay_map


class DiffusedFirstVisitDelay(DelayMethod):
    """FirstVisitDelay values BFS-propagated to all reachable cells.

    Addresses the sparsity of FirstVisitDelay: instead of only marking cells
    on the agent's CBS path, the delay signal is spread outward so that LaCAM
    receives guidance even when exploring off-path configurations.

    Propagation uses a max-priority Dijkstra: each cell receives the highest
    delay that reaches it, decayed by `decay` per hop. Only values above
    `min_value` are propagated further.

    Args:
        decay:     Multiplicative factor per hop (default 0.7). Lower = faster
                   decay, tighter signal. Higher = wider spread.
        min_value: Propagation cutoff — values below this are dropped.
    """

    def __init__(self, decay: float = 0.7, min_value: float = 0.05):
        self.decay = decay
        self.min_value = min_value

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        seed = FirstVisitDelay().compute(grid, bfs_cache, paths, goals, agent_idx)
        return _diffuse(grid, seed, self.decay, self.min_value)


class GlobalCongestionDelay(DelayMethod):
    """Delay based on weighted temporal congestion from all other agents' CBS paths.

    For each cell v on agent i's path (first visited at time t_i):
        delay_i(v) = sum over agents j != i that also visit v:
                         1 / (1 + |t_j_first_visit(v) - t_i_first_visit(v)|)

    Agents visiting v at the same time as agent i contribute 1.0 each;
    agents visiting further away in time contribute less. There is no hard
    cutoff — contributions decay smoothly with temporal distance.
    Cells not on agent i's path keep delay = 0.
    """

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:  # noqa: ARG002
        first_visits: list[dict[Coord, int]] = []
        for path in paths:
            fv: dict[Coord, int] = {}
            for t, coord in enumerate(path):
                if coord not in fv:
                    fv[coord] = t
            first_visits.append(fv)

        delay_map = np.zeros(grid.shape, dtype=np.float32)

        for coord, t_self in first_visits[agent_idx].items():
            congestion = 0.0
            for j, fv_j in enumerate(first_visits):
                if j == agent_idx:
                    continue
                if coord in fv_j:
                    congestion += 1.0 / (1.0 + abs(fv_j[coord] - t_self))
            delay_map[coord] = congestion

        return delay_map


class DiffusedGlobalCongestionDelay(DelayMethod):
    """GlobalCongestionDelay values BFS-propagated to all reachable cells.

    Addresses the sparsity of GlobalCongestionDelay: congestion scores on the
    agent's CBS path are spread outward so cells near congested regions also
    receive elevated delay, even if the agent never visits them directly.

    Propagation uses the same max-priority Dijkstra as the other Diffused*
    variants: each cell receives the highest congestion value that reaches it,
    decayed by `decay` per hop.

    Args:
        decay:     Multiplicative factor per hop (default 0.7).
        min_value: Propagation cutoff.
    """

    def __init__(self, decay: float = 0.7, min_value: float = 0.05):
        self.decay = decay
        self.min_value = min_value

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        seed = GlobalCongestionDelay().compute(grid, bfs_cache, paths, goals, agent_idx)
        scaling_factor = 0.5
        return _diffuse(grid, seed, self.decay, self.min_value) * scaling_factor


class WaitTimeDelay(DelayMethod):
    """Delay = total timesteps any agent waits at each cell in the CBS solution.

    A "wait" is any timestep where an agent stays in the same cell as the
    previous step. Unlike FirstVisitDelay (which captures the full cost-to-go
    residual), this isolates the literal waiting overhead: if CBS made an agent
    wait k steps at v, the cell gets delay += k.

    Sparsest of all methods — only cells where CBS actually caused waiting get
    nonzero delay — but the signal is integer-valued and unambiguous.
    Use DiffusedWaitTimeDelay if coverage is too sparse for your map.
    """

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:  # noqa: ARG002
        delay_map = np.zeros(grid.shape, dtype=np.float32)
        for path in paths:
            for t in range(1, len(path)):
                if path[t] == path[t - 1]:
                    delay_map[path[t]] += 1.0
        return delay_map


class DiffusedWaitTimeDelay(DelayMethod):
    """WaitTimeDelay values BFS-propagated to all reachable cells.

    Same propagation as DiffusedFirstVisitDelay but seeded from WaitTimeDelay.
    Good starting point if WaitTimeDelay is too sparse and FirstVisitDelay
    signal is too noisy.

    Args:
        decay:     Multiplicative factor per hop (default 0.7).
        min_value: Propagation cutoff.
    """

    def __init__(self, decay: float = 0.7, min_value: float = 0.05):
        self.decay = decay
        self.min_value = min_value

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        seed = WaitTimeDelay().compute(grid, bfs_cache, paths, goals, agent_idx)
        return _diffuse(grid, seed, self.decay, self.min_value)


class GoalPressureDelay(DelayMethod):
    """Delay = sum of BFS-proximity contributions from all other agents' goals.

    For each free cell v:
        delay(v) = sum over agents j != i:  1 / (1 + bfs(goal_j, v))

    Cells close to many other agents' goals receive high delay, steering agent i
    away from convergence hot-spots. Does not require CBS paths — works from
    map + goals alone, so it can be used even without a CBS binary.
    """

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        delay_map = np.zeros(grid.shape, dtype=np.float32)
        for j, goal in enumerate(goals):
            if j == agent_idx:
                continue
            bfs = bfs_cache[goal]
            # bfs value is grid.size (very large) for obstacles → contribution ≈ 0
            delay_map += 1.0 / (1.0 + bfs)
        delay_map[~grid] = 0.0
        return delay_map


class TopologicalDelay(DelayMethod):
    """Delay based on local map topology — corridors and dead-ends get higher delay.

    A free cell with k passable neighbors gets:
        delay(v) = (4 - k) * scale

    Dead-end (1 neighbour):  delay = 3 * scale
    Corridor  (2 neighbours): delay = 2 * scale
    T-junction (3 neighbours): delay = 1 * scale
    Open space (4 neighbours): delay = 0

    Pure map signal — no paths, no BFS, no CBS needed. Particularly effective
    on warehouse and maze-like maps where bottlenecks are structural. Combine
    with DiffusedFirstVisitDelay for richer coverage on open maps.

    Args:
        scale: Multiplier on the (4 - degree) penalty (default 1.0).
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:  # noqa: ARG002
        delay_map = np.zeros(grid.shape, dtype=np.float32)
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x]:
                    n = len(get_neighbors(grid, (y, x)))
                    delay_map[y, x] = max(0.0, (4 - n) * self.scale)
        return delay_map


class RandomDelay(DelayMethod):
    """Random delay values for each free cell.

    Each free cell gets a random value in [0, scale]. Obstacles get 0.
    Purely stochastic — no paths, no BFS, no CBS needed. Useful for testing
    robustness of LaCAM to noisy heuristics.

    Args:
        scale: Maximum random delay (default 1.0).
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:  # noqa: ARG002
        delay_map = np.random.uniform(0.0, self.scale, size=grid.shape).astype(
            np.float32
        )
        delay_map[~grid] = 0.0
        return delay_map


# ── Shared helper ─────────────────────────────────────────────────────────────


def _diffuse(
    grid: Grid,
    seed_map: np.ndarray,
    decay: float,
    min_value: float,
) -> np.ndarray:
    """Max-priority Dijkstra propagation of delay values across free cells.

    Each cell receives the highest delay that reaches it; that value decays
    by `decay` per hop. Cells seeded with nonzero values in `seed_map` are
    the starting points.
    """
    delay = seed_map.copy()
    # Max-heap via negated values: (-delay, y, x)
    heap: list[tuple[float, int, int]] = [
        (-float(seed_map[y, x]), y, x)
        for y in range(grid.shape[0])
        for x in range(grid.shape[1])
        if grid[y, x] and seed_map[y, x] > 0
    ]
    heapq.heapify(heap)

    while heap:
        neg_d, y, x = heapq.heappop(heap)
        d = -neg_d
        if d < delay[y, x] - 1e-9:  # stale entry
            continue
        propagated = d * decay
        if propagated < min_value:
            continue
        for vy, vx in get_neighbors(grid, (y, x)):
            if propagated > delay[vy, vx]:
                delay[vy, vx] = propagated
                heapq.heappush(heap, (-propagated, vy, vx))

    return delay


# ── Registry ──────────────────────────────────────────────────────────────────
# Add new methods here and they become available via --delay-method.

DELAY_METHODS: dict[str, type[DelayMethod]] = {
    "zero": ZeroDelay,
    "first_visit": FirstVisitDelay,
    "diffused_first_visit": DiffusedFirstVisitDelay,
    "global_congestion": GlobalCongestionDelay,
    "diffused_global_congestion": DiffusedGlobalCongestionDelay,
    "wait_time": WaitTimeDelay,
    "diffused_wait_time": DiffusedWaitTimeDelay,
    "goal_pressure": GoalPressureDelay,
    "topological": TopologicalDelay,
    "random": RandomDelay,
    "non_optimal_penalty": NonOptimalPenaltyDelay,
    "non_astar_penalty": NonAStarPenaltyDelay,
}


def get_delay_method(name: str) -> DelayMethod:
    if name not in DELAY_METHODS:
        raise ValueError(
            f"Unknown delay method '{name}'. Choose from: {list(DELAY_METHODS)}"
        )
    return DELAY_METHODS[name]()
