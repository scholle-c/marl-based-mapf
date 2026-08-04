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
from collections import deque
from collections.abc import Callable

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


class NonOptimalBigPenaltyDelay(DelayMethod):
    """See NonOptimalPenaltyDelay, but with a bigger penalty (e.g. +100000) for each cell that isnt on the optimal path of the agent."""

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        path = paths[agent_idx]
        delay_map = np.ones(grid.shape, dtype=np.float32) * 100000
        if not path:
            return delay_map

        for coord in path:
            delay_map[coord] = 0.0  # No penalty for cells on the optimal path

        return delay_map


class NonOptimalPenaltyBFSDelay(DelayMethod):
    """Delay = Penalty (e.g. x + 0 if on optimal path, x = min(neigh(x)) + 1 otherwise) for each cell that isnt on the optimal path of the agent.

    I want to check if the heuristic works better than the NonOptimalPenaltyDelay (where the penalty is constant at 1),
    because it encourages the agent to get back on track faster if it is further away from the optimal path.
    """

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        path = paths[agent_idx]
        max_value = np.max(grid.shape) * 2

        delay_map = np.ones(grid.shape, dtype=np.float32) * max_value
        if not path:
            return delay_map

        # Multi-source BFS from the path cells, propagating +1 penalty per hop.
        visited = set(path)
        for coord in path:
            delay_map[coord] = 0.0  # No penalty for cells on the optimal path

        open_queue: deque = deque(visited)
        while open_queue:
            current = open_queue.popleft()
            current_delay = delay_map[current]
            for neighbor in get_neighbors(grid, current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    delay_map[neighbor] = current_delay + 1
                    open_queue.append(neighbor)

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


class CBSPath(DelayMethod):
    """Signal = calculated on the fly using the CBS path: The Get method returns the distance to goal + a signal towards the CBS path.
    Whats new is, that this method considers time --> Waiting and revisiting of cells is considered in heuristic/Guiding.

    Right now this method is extreme in punishing the agent for not being on the CBS path (+100000) to see, if this can close the gap completly
    """

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        # paths is a nested list: outer dim = agents, inner dim = coordinates
        # visited over time. Pad it into a dense (n_agents, max_len, 2) array,
        # filling shorter paths with each agent's last coordinate so all agents
        # share the same time dimension.
        if not paths:
            return np.zeros((0, 0, 2), dtype=np.int32)

        max_len = max(len(path) for path in paths)
        padded = np.zeros((len(paths), max_len, 2), dtype=np.int32)
        for i, path in enumerate(paths):
            if not path:
                continue
            padded[i, : len(path)] = path
            if len(path) < max_len:
                padded[i, len(path) :] = path[-1]

        return padded[agent_idx]


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


class SpaceTimeDelay(DelayMethod):
    """Delay = true cost-to-go with the other agents treated as *moving* obstacles.

    Unlike the path-marking methods, this fills every cell with a physically
    meaningful value: "if I were on cell v (at planning time), following the
    optimal single-agent route while every *other* agent walks its fixed CBS
    path, how many steps do I still need?" — minus the plain BFS distance.

        delay(v) = d(v, t0) - h_bfs(v)

    d(v, t) is computed by a backward dynamic program over the time-expanded
    grid (state = (cell, time)):

        d(g, t)  = 0
        d(v, t)  = 1 + min_{u in neighbours(v) + {v}} d(u, t+1)

    subject to, for the move v -> u at step t -> t+1:
        * vertex:  u is not occupied by another agent at t+1
        * swap:    no other agent traverses the edge u -> v at t -> t+1

    Other agents follow paths[j]; once their path ends they stay parked on
    their final cell (a permanent obstacle). The world becomes static after
    T = max_j (len(path_j) - 1), so the top layer d(., T) is a plain backward
    BFS from the goal with the parked cells blocked, and layers T-1 .. 0 are
    swept in one pass each (each layer depends only on layer t+1).

    Collapse to the 2D map required by the heuristic:
        * "t0"  (default): d(v, 0) — cost-to-go if you were at v *now*. Cells
                 blocked at t=0 (or unreachable) fall back to delay 0.
        * "min": min_t d(v, t) — best-case cost-to-go over all times. Ignores
                 the "occupied right now" artefact; keeps only structural /
                 persistent congestion. Useful as an ablation.

    Congestion model:
        * park_blocks=False (default, "transient"): an agent is an obstacle only
                 while it is still moving along its path; once it reaches its
                 goal it is ignored and the map beyond the horizon is assumed
                 clear (top layer = plain BFS). Delay then measures *only* the
                 waiting/detour forced by crossing traffic — small and sparse.
        * park_blocks=True ("permanent"): every other agent also blocks its goal
                 cell forever (top layer = BFS with parked cells removed). Far
                 more pessimistic; as a *static* heuristic it tends to swamp the
                 BFS term on dense maps. Kept for ablation.

    Args:
        collapse:    "t0" or "min" (default "t0").
        park_blocks: model parked agents as permanent obstacles (default False).
    """

    def __init__(self, collapse: str = "t0", park_blocks: bool = False):
        if collapse not in ("t0", "min"):
            raise ValueError(f"collapse must be 't0' or 'min', got {collapse!r}")
        self.collapse = collapse
        self.park_blocks = park_blocks

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        delay_map = np.zeros(grid.shape, dtype=np.float32)
        if not paths[agent_idx]:
            return delay_map

        goal = goals[agent_idx]
        others = [j for j in range(len(paths)) if j != agent_idx and paths[j]]

        def pos(j: int, t: int) -> Coord | None:
            """Cell of agent j at time t. None once it has parked (transient mode)."""
            p = paths[j]
            if t < len(p):
                return p[t]
            return p[-1] if self.park_blocks else None

        # Horizon after which every other agent has finished its path.
        T = max((len(paths[j]) - 1 for j in others), default=0)

        # Per-timestep occupancy and traversed edges of the other agents.
        occ: list[set[Coord]] = [
            {c for j in others if (c := pos(j, t)) is not None} for t in range(T + 1)
        ]
        edges: list[set[tuple[Coord, Coord]]] = [
            {
                (a, b)
                for j in others
                if (a := pos(j, t)) is not None and (b := pos(j, t + 1)) is not None
            }
            for t in range(T)
        ]

        # Top layer d(., T): cost-to-go once the horizon is over.
        if self.park_blocks:
            d_next = _static_cost_to_go(grid, goal, occ[T] - {goal})
        else:
            # Map assumed clear beyond the horizon → plain BFS distance.
            d_next = np.where(bfs_cache[goal] < bfs_cache.NIL, bfs_cache[goal], np.inf)
            d_next = d_next.astype(np.float32)
        d_min = d_next.copy()

        free_cells = [
            (y, x)
            for y in range(grid.shape[0])
            for x in range(grid.shape[1])
            if grid[y, x]
        ]

        # Backward sweep: layer t depends only on layer t+1.
        for t in range(T - 1, -1, -1):
            occ_t = occ[t]
            occ_t1 = occ[t + 1]
            edges_t = edges[t]
            d_cur = np.full(grid.shape, np.inf, dtype=np.float32)
            for v in free_cells:
                if v in occ_t:  # cannot stand here at time t
                    continue
                if v == goal:
                    d_cur[v] = 0.0
                    continue
                best = np.inf
                for u in get_neighbors(grid, v) + [v]:  # moves + wait
                    if u in occ_t1:  # vertex collision at t+1
                        continue
                    if (u, v) in edges_t:  # head-on swap
                        continue
                    cand = 1.0 + d_next[u]
                    if cand < best:
                        best = cand
                d_cur[v] = best
            d_next = d_cur
            np.minimum(d_min, d_cur, out=d_min)

        d_final = d_next if self.collapse == "t0" else d_min

        bfs_table = bfs_cache[goal]
        cap = float(grid.size)
        for v in free_cells:
            d = d_final[v]
            h = bfs_table[v]
            if not np.isfinite(d) or h >= bfs_cache.NIL:
                continue  # blocked/unreachable → trust plain BFS (delay 0)
            delay_map[v] = min(cap, max(0.0, float(d - h)))

        return delay_map


class CbsFunnelDelay(DelayMethod):
    """Dense, smooth funnel that channels an agent onto its CBS path.

    This is the *guidance* counterpart to the cost-to-go methods: instead of
    estimating true remaining cost (which rewards selfish deviation from the
    coordinated plan), it simply makes every cell more expensive the further it
    lies from the agent's CBS path, so a greedy descent slides back onto it.

        funnel(v) = geodesic hop-distance from v to the nearest CBS-path cell

    On the path funnel = 0; it grows by 1 per step away, obstacles routed
    around (true graph distance, multi-source BFS). Added on top of BFS this
    gives h_total(v) = dist_to_goal(v) + dist_to_path(v): the agent is pulled
    toward the goal *and* toward the coordinated corridor.

    This is the intended learning target (a model predicts it from map + agent
    endpoints, without ever running CBS at inference). It is a denser, smoother
    version of the binary NonOptimalPenaltyDelay: every cell is supervised, and
    because a funnel is robust to which of several equally-optimal CBS paths was
    picked (averaging two funnels ≈ a valid, slightly wider funnel — unlike
    averaging two binary paths, which blurs into a broken path), it carries far
    less tie-break label noise than the binary target.

    Args:
        tau:       If set, saturate via 1 - exp(-d/tau) → bounded in [0, 1),
                   smoother far field. If None (default), raw linear hops —
                   directly injectable into LaCAM (same scale as BFS steps).
        normalize: Divide by the per-map max so the target lives in [0, 1].
                   Convenient for NN regression; do NOT combine with LaCAM
                   injection (kills the scale relative to h_bfs).
    """

    def __init__(self, tau: float | None = None, normalize: bool = False):
        self.tau = tau
        self.normalize = normalize

    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:  # noqa: ARG002
        path = paths[agent_idx]
        if not path:
            return np.zeros(grid.shape, dtype=np.float32)

        # Multi-source BFS: hop-distance from every free cell to the CBS path.
        dist = np.full(grid.shape, np.inf, dtype=np.float32)
        Q: deque = deque()
        for c in path:
            if dist[c] != 0.0:
                dist[c] = 0.0
                Q.append(c)
        while Q:
            u = Q.popleft()
            d = dist[u]
            for v in get_neighbors(grid, u):
                if d + 1 < dist[v]:
                    dist[v] = d + 1
                    Q.append(v)

        funnel = dist
        funnel[~np.isfinite(funnel)] = 0.0  # disconnected → no guidance
        if self.tau is not None:
            funnel = (1.0 - np.exp(-funnel / self.tau)).astype(np.float32)
        if self.normalize:
            m = float(funnel.max())
            if m > 0.0:
                funnel = funnel / m
        funnel[~grid] = 0.0
        return funnel.astype(np.float32)


# ── Shared helper ─────────────────────────────────────────────────────────────


def _static_cost_to_go(grid: Grid, goal: Coord, blocked: set[Coord]) -> np.ndarray:
    """Backward BFS from `goal`, treating `blocked` cells as extra obstacles.

    Returns a float map of steps-to-goal; unreachable cells are +inf. The goal
    itself is always reachable (0), even if another agent is parked on it.
    """
    table = np.full(grid.shape, np.inf, dtype=np.float32)
    table[goal] = 0.0
    Q: deque = deque([goal])
    while Q:
        u = Q.popleft()
        d = table[u]
        for v in get_neighbors(grid, u):
            if v in blocked:
                continue
            if d + 1 < table[v]:
                table[v] = d + 1
                Q.append(v)
    return table


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

DELAY_METHODS: dict[str, Callable[[], DelayMethod]] = {
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
    "non_optimal_big_penalty": NonOptimalBigPenaltyDelay,
    "cbs_funnel": NonOptimalPenaltyBFSDelay,
    "non_astar_penalty": NonAStarPenaltyDelay,
    "space_time": SpaceTimeDelay,  # transient congestion, t0 collapse (default)
    "space_time_parked": lambda: SpaceTimeDelay(park_blocks=True),
    "cbs_funnel": CbsFunnelDelay,  # dense guidance funnel (learning target)
    "cbs_funnel_sat": lambda: CbsFunnelDelay(tau=3.0),
    "cbs_path": CBSPath,
}


def get_delay_method(name: str) -> DelayMethod:
    if name not in DELAY_METHODS:
        raise ValueError(
            f"Unknown delay method '{name}'. Choose from: {list(DELAY_METHODS)}"
        )
    return DELAY_METHODS[name]()
