from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from .dist_table import BfsCache, DistTable
from marl_path.shared.mapf_utils import (
    Config,
    Configs,
    Coord,
    Deadline,
    Grid,
    get_neighbors,
)
from .pibt import PIBT


@dataclass
class LowLevelNode:
    who: list[int] = field(default_factory=lambda: [])
    where: list[Coord] = field(default_factory=lambda: [])
    depth: int = 0

    def get_child(self, who: int, where: Coord) -> LowLevelNode:
        return LowLevelNode(
            who=self.who + [who],
            where=self.where + [where],
            depth=self.depth + 1,
        )


@dataclass
class HighLevelNode:
    Q: Config
    order: list[int]
    parent: HighLevelNode | None = None
    tree: deque[LowLevelNode] = field(default_factory=lambda: deque([LowLevelNode()]))
    g: int = 0
    h: int = 0
    f: int = g + h
    neighbors: set[HighLevelNode] = field(default_factory=lambda: set())

    def __eq__(self, other) -> bool:
        if isinstance(other, HighLevelNode):
            return self.Q == other.Q
        return False

    def __hash__(self) -> int:
        return self.Q.__hash__()


class LaCAM:
    def __init__(self) -> None:
        pass

    def solve(
        self,
        grid: Grid,
        starts: Config,
        goals: Config,
        delay_maps: Optional[list[np.ndarray]] = None,
        delay_method: str = "max",
        cbs_path_penalty: float = 100000.0,
        time_limit_ms: int = 3000,
        deadline: Optional[Deadline] = None,
        flg_star: bool = True,
        seed: int = 0,
        verbose: int = 1,
    ) -> Configs:
        self.num_agents: int = len(starts)
        self.grid: Grid = grid
        self.starts: Config = starts
        self.goals: Config = goals
        self.delay_maps = delay_maps
        self.delay_method = delay_method
        self.cbs_path_penalty = cbs_path_penalty
        self.deadline: Deadline = (
            deadline if deadline is not None else Deadline(time_limit_ms)
        )
        self.flg_star: bool = flg_star
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)
        self.verbose = verbose
        return self._solve()

    def _solve(self) -> Configs:
        self.info(1, "start solving MAPF")

        self.bfs_cache: BfsCache = BfsCache(self.grid)
        self.dist_tables: list[DistTable] = []
        for i, g in enumerate(self.goals):
            delay_map = self.delay_maps[i] if self.delay_maps is not None else None
            self.dist_tables.append(DistTable(
                self.grid, g,
                bfs_cache=self.bfs_cache,
                delay_map=delay_map,
                delay_method=self.delay_method,
                cbs_path_penalty=self.cbs_path_penalty,
            ))
        self.pibt = PIBT(self.dist_tables)

        OPEN: deque[HighLevelNode] = deque([])
        EXPLORED: dict[Config, HighLevelNode] = {}
        N_goal: HighLevelNode | None = None

        Q_init = self.starts
        N_init = HighLevelNode(
            Q=Q_init, order=self.get_order(Q_init), h=self.get_h_value(Q_init)
        )
        OPEN.appendleft(N_init)
        EXPLORED[N_init.Q] = N_init

        while len(OPEN) > 0 and not self.deadline.is_expired:
            N: HighLevelNode = OPEN[0]
            timestep = self.get_timestep(N)

            if N_goal is None and N.Q == self.goals:
                N_goal = N
                self.info(1, f"initial solution found, cost={N_goal.g}")
                if not self.flg_star:
                    break

            if N_goal is not None and N_goal.g <= N.f:
                OPEN.popleft()
                continue

            if len(N.tree) == 0:
                OPEN.popleft()
                continue

            C: LowLevelNode = N.tree.popleft()
            if C.depth < self.num_agents:
                i = N.order[C.depth]
                v = N.Q[i]
                cands = [v] + get_neighbors(self.grid, v)
                self.rng.shuffle(cands)
                for u in cands:
                    N.tree.append(C.get_child(i, u))

            Q_to = self.configuration_generaotr(N, C)
            if Q_to is None:
                continue
            elif Q_to in EXPLORED.keys():
                N_known = EXPLORED[Q_to]
                N.neighbors.add(N_known)
                OPEN.appendleft(N_known)
                D = deque([N])
                while len(D) > 0 and self.flg_star:
                    N_from = D.popleft()
                    for N_to in N_from.neighbors:
                        g = N_from.g + self.get_edge_cost(N_from.Q, N_to.Q)
                        if g < N_to.g:
                            if N_goal is not None and N_to is N_goal:
                                self.info(2, f"cost update: {N_goal.g:4d} -> {g:4d}")
                            N_to.g = g
                            N_to.f = N_to.g + N_to.h
                            N_to.parent = N_from
                            D.append(N_to)
                            if N_goal is not None and N_to.f < N_goal.g:
                                OPEN.appendleft(N_to)
            else:
                N_new = HighLevelNode(
                    Q=Q_to,
                    parent=N,
                    order=self.get_order(Q_to),
                    g=N.g + self.get_edge_cost(N.Q, Q_to),
                    h=self.get_h_value(Q_to),
                )
                N.neighbors.add(N_new)
                OPEN.appendleft(N_new)
                EXPLORED[Q_to] = N_new

        if N_goal is not None and len(OPEN) == 0:
            self.info(1, f"reach optimal solution, cost={N_goal.g}")
        elif N_goal is not None:
            self.info(1, f"suboptimal solution, cost={N_goal.g}")
        elif len(OPEN) == 0:
            self.info(1, "detected unsolvable instance")
        else:
            self.info(1, "failure due to timeout")
        return self.backtrack(N_goal)

    @staticmethod
    def backtrack(_N: HighLevelNode | None) -> Configs:
        configs: Configs = []
        N = _N
        while N is not None:
            configs.append(N.Q)
            N = N.parent
        configs.reverse()
        return configs

    @staticmethod
    def get_timestep(_N: HighLevelNode) -> int:
        timestep: int = 0
        N = _N
        while N is not None:
            timestep += 1
            N = N.parent
        return timestep

    def get_edge_cost(self, Q_from: Config, Q_to: Config) -> int:
        cost = 0
        for i in range(self.num_agents):
            if not (self.goals[i] == Q_from[i] == Q_to[i]):
                cost += 1
        return cost

    def get_h_value(self, Q: Config) -> int:
        CONSIDER_DELAY = False

        cost = 0
        for agent_idx, loc in enumerate(Q):
            if CONSIDER_DELAY:
                c = self.dist_tables[agent_idx].get(loc)
            else:
                c = self.dist_tables[agent_idx].table[loc]
            if c is None:
                return np.iinfo(np.int32).max
            cost += c
        return cost

    def get_order(self, Q: Config) -> list[int]:
        # Priority is based on raw BFS distance-to-goal, not the
        # delay-augmented distance — delay is meant to steer PIBT's move
        # choice (via DistTable.get) and the high-level heuristic (h), not
        # who gets first pick of a cell this timestep.
        CONSIDER_DELAY = False

        order = list(range(self.num_agents))
        self.rng.shuffle(order)
        if CONSIDER_DELAY:
            order.sort(
                key=lambda i: self.dist_tables[i].get(Q[i]), reverse=True
            )
        else:
            order.sort(key=lambda i: self.dist_tables[i].table[Q[i]], reverse=True)
        return order

    def configuration_generaotr(
        self, N: HighLevelNode, C: LowLevelNode
    ) -> Config | None:
        Q_to = Config([self.pibt.NIL_COORD for _ in range(self.num_agents)])
        for k in range(C.depth):
            Q_to[C.who[k]] = C.where[k]

        timestep = self.get_timestep(N)
        success = self.pibt.step(N.Q, Q_to, N.order, timestep)
        return Q_to if success else None

    def info(self, level: int, msg: str) -> None:
        if self.verbose < level:
            return
        logger.debug(f"{int(self.deadline.elapsed):4d}ms  {msg}")
