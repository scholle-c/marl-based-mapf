"""Structure-preserving noise operators on CBS paths.

Used to answer: *how exact must a learned path be before the time-indexed
heuristic stops closing the gap?*

Design constraint — every operator must return a **valid walk**: same start
cell, connected, only grid-legal moves (step to a neighbour or wait). Naive
i.i.d. cell-level noise is deliberately not offered here, for two reasons:

1. It destroys path connectivity, and hypothesis #21 showed that ~54% of the
   heuristic's benefit comes from pure path *consistency* — so it would
   measure path destruction, not model error.
2. An autoregressive model rolling out from a known start over a 5-action
   alphabet cannot produce a disconnected path in the first place, so it is
   an error mode the target architecture never exhibits.

Operators are selected by name via `apply_path_noise`. Currently implemented:

    none        no-op
    wait_shift  (Operator B) insert/delete wait actions — spatially identical,
                only the timing moves. Isolates temporal precision.

Planned extension points (same signature, register in PATH_NOISE_OPS):
    detour      (Operator A) reroute a segment between the same endpoints
    action      (Operator C) random valid action with prob. p, then re-roll out
"""

from __future__ import annotations

import numpy as np

from marl_path.shared.mapf_utils import BfsCache, Coord, Grid, get_neighbors

Path = list[Coord]


def _wait_indices(path: Path) -> list[int]:
    """Indices t where the agent waits, i.e. path[t] == path[t + 1]."""
    return [t for t in range(len(path) - 1) if path[t] == path[t + 1]]


def wait_shift(path: Path, rng: np.random.Generator, ops: int = 1) -> Path:
    """Insert or delete `ops` wait actions.

    Deleting a wait pulls everything after it one timestep earlier, inserting
    one pushes it one later. The visited cells and their order are unchanged —
    only *when* the agent is where moves. Start and goal are preserved.
    """
    if len(path) < 2:
        return list(path)

    out = list(path)
    for _ in range(ops):
        waits = _wait_indices(out)
        # Delete only if a wait exists; otherwise we can always insert one.
        delete = len(waits) > 0 and rng.random() < 0.5
        if delete:
            t = int(rng.choice(waits))
            del out[t + 1]
        else:
            t = int(rng.integers(0, len(out)))
            out.insert(t, out[t])
        if len(out) < 2:
            break
    return out


def _action_noise_impl(
    path: Path,
    rng: np.random.Generator,
    ctx: "NoiseContext | None",
    p: float,
    recovery: str,
) -> Path:
    """Re-roll the path, taking a wrong action with probability p.

    This is the only operator family with **error compounding**: a wrong action
    moves the agent off the CBS path, and everything after that is planned from
    the new position — exactly how an autoregressive policy with per-step error
    rate p derails.

    `recovery` decides what an off-track agent does:
      "goal" — descend the BFS distance to its goal (greedy shortest path,
               discarding the CBS plan; pessimistic)
      "path" — head for the next CBS waypoint (keeps the plan; optimistic)

    Either way the agent resumes following the CBS path if it lands back on it,
    and only ever rejoins at an index ahead of its current progress.

    Unlike `wait_shift` this changes the *route*, so `time_displacement` does
    not apply — use `divergence_stats` for these operators.
    """
    if len(path) < 2 or ctx is None:
        return list(path)

    grid, goal_dist = ctx.grid, ctx.goal_dist

    # All occurrences per cell — CBS paths revisit cells (waits, crossings), so
    # a first-occurrence map would rewind the agent and make it oscillate.
    occurrences: dict[Coord, list[int]] = {}
    for i, cell in enumerate(path):
        occurrences.setdefault(cell, []).append(i)

    goal = path[-1]
    out: Path = [path[0]]
    cur = path[0]
    k: int | None = 0  # index in the true path, or None while off-track
    progress = 0  # highest path index reached, so rejoins only ever move forward
    budget = 3 * len(path) + 10

    while budget > 0:
        if k is not None and k >= len(path) - 1:
            break
        if k is None and cur == goal:
            break
        budget -= 1

        if rng.random() < p:
            options = [cur, *get_neighbors(grid, cur)]
            cur = options[int(rng.integers(0, len(options)))]
            k = None
        elif k is not None:
            # On track: follow the CBS path. This branch alone runs when p == 0,
            # which makes the operator exactly identity-preserving.
            k += 1
            progress = k
            cur = path[k]
            out.append(cur)
            continue
        else:
            nbrs = get_neighbors(grid, cur)
            if not nbrs:
                break
            if recovery == "path":
                # Steer back to the next CBS waypoint instead of to the goal.
                target = path[min(progress + 1, len(path) - 1)]
                dist = ctx.bfs_cache[target]
            else:
                dist = goal_dist
            cur = min(nbrs, key=lambda n: dist[n])

        out.append(cur)
        # Off track — rejoin only at an occurrence strictly ahead of where we
        # already got to, otherwise we would travel backwards in time.
        ahead = [i for i in occurrences.get(cur, ()) if i > progress]
        if ahead:
            k = ahead[0]
            progress = k

    return out


def action_noise_rejoin(
    path: Path,
    rng: np.random.Generator,
    ops: int,
    ctx: "NoiseContext | None" = None,
    p: float = 0.05,
) -> Path:
    """(Operator C, variant) Like `action_noise`, but recovers toward the CBS path.

    `action_noise` makes a deviating agent fall back to a greedy shortest path
    to its goal — i.e. exactly the A*-style route that hypothesis #4 showed is
    a much worse target than the CBS route. That is a modelling choice, not a
    given, and it likely makes the operator pessimistic: a model conditioned on
    the whole instance would steer back toward its CBS plan rather than discard
    it.

    This variant does that instead: after a wrong action the agent heads for the
    next CBS waypoint. The two operators bracket the range the real requirement
    lies in.
    """
    return _action_noise_impl(path, rng, ctx, p, recovery="path")


def action_noise(
    path: Path,
    rng: np.random.Generator,
    ops: int,  # noqa: ARG001 — parameterised by `p`, not op count
    ctx: "NoiseContext | None" = None,
    p: float = 0.05,
) -> Path:
    """(Operator C) Re-roll the path, taking a wrong action with probability p.

    See `_action_noise_impl`. After a deviation the agent falls back to a greedy
    shortest path to its goal — the pessimistic end of the bracket. Compare
    `action_noise_rejoin`.
    """
    return _action_noise_impl(path, rng, ctx, p, recovery="goal")


class NoiseContext:
    """Per-agent context some operators need (grid topology, distance tables)."""

    __slots__ = ("grid", "goal_dist", "bfs_cache")

    def __init__(self, grid: Grid, goal_dist: np.ndarray, bfs_cache: BfsCache):
        self.grid = grid
        self.goal_dist = goal_dist
        self.bfs_cache = bfs_cache


PATH_NOISE_OPS: dict[str, object] = {
    "none": None,
    "wait_shift": wait_shift,
    "action": action_noise,
    "action_rejoin": action_noise_rejoin,
}

# Operators that change the route and therefore need `divergence_stats`
# rather than `time_displacement`.
ROUTE_CHANGING = {"action", "action_rejoin"}


def _route_arrivals(path: Path) -> list[tuple[Coord, int]]:
    """Collapse waits: the sequence of distinct cells with their arrival time.

    `wait_shift` only ever adds or removes waits, so the *route* (which cells,
    in which order) is invariant under it — only the arrival times move.
    """
    out: list[tuple[Coord, int]] = []
    for t, cell in enumerate(path):
        if not out or out[-1][0] != cell:
            out.append((cell, t))
    return out


def time_displacement(true_path: Path, noisy_path: Path) -> tuple[float, int]:
    """(mean, max) absolute arrival-time shift along the shared route.

    This is the *realized* error magnitude, which is what should be reported
    on the x-axis of a noise sweep. The nominal op count is not a usable
    measure: `wait_shift` draws insert-or-delete per op, so two ops cancel
    roughly half the time and the axis is non-monotone.

    Returns (0.0, 0) if the routes differ, which `wait_shift` never causes.
    """
    a, b = _route_arrivals(true_path), _route_arrivals(noisy_path)
    if len(a) != len(b) or any(x[0] != y[0] for x, y in zip(a, b)):
        return 0.0, 0
    deltas = [abs(x[1] - y[1]) for x, y in zip(a, b)]
    if not deltas:
        return 0.0, 0
    return sum(deltas) / len(deltas), max(deltas)


def displacement_stats(
    true_paths: list[Path], noisy_paths: list[Path]
) -> tuple[float, int]:
    """(mean, max) arrival-time shift aggregated over all agents.

    The mean is taken over *all* agents, including untouched ones, so it scales
    with both the fraction of agents hit and the per-agent error size.
    """
    if not true_paths:
        return 0.0, 0
    per_agent = [time_displacement(t, n) for t, n in zip(true_paths, noisy_paths)]
    means = [m for m, _ in per_agent]
    maxes = [x for _, x in per_agent]
    return sum(means) / len(means), max(maxes) if maxes else 0


def apply_path_noise(
    paths: list[Path],
    noise: str,
    level: float,
    ops: int = 1,
    seed: int = 0,
    p: float = 0.05,
    grid: Grid | None = None,
    bfs_cache: BfsCache | None = None,
    goals: list[Coord] | None = None,
) -> list[Path]:
    """Return a perturbed copy of `paths`.

    Args:
        paths: CBS paths, one per agent.
        noise: operator name from PATH_NOISE_OPS.
        level: fraction of agents perturbed (0.0-1.0).
        ops:   perturbations applied per affected agent (`wait_shift`).
        seed:  RNG seed for agent selection and the operator itself.
        p:     per-step error probability (`action`).
        grid, bfs_cache, goals: required by route-changing operators.

    The input list is never mutated — soc_cbs must stay based on the true
    CBS paths.
    """
    if noise == "none" or level <= 0.0 or not paths:
        return [list(p) for p in paths]

    if noise not in PATH_NOISE_OPS:
        raise ValueError(
            f"Unknown path noise '{noise}'. Choose from: {list(PATH_NOISE_OPS)}"
        )
    op = PATH_NOISE_OPS[noise]
    assert op is not None

    needs_ctx = noise in ROUTE_CHANGING
    if needs_ctx and (grid is None or bfs_cache is None or goals is None):
        raise ValueError(f"Path noise '{noise}' requires grid, bfs_cache and goals.")

    rng = np.random.default_rng(seed)
    n_affected = int(round(level * len(paths)))
    if n_affected <= 0:
        return [list(p) for p in paths]

    affected = set(
        rng.choice(len(paths), size=min(n_affected, len(paths)), replace=False).tolist()
    )

    out: list[Path] = []
    for i, path in enumerate(paths):
        if i not in affected:
            out.append(list(path))
            continue
        if needs_ctx:
            assert grid is not None and bfs_cache is not None and goals is not None
            ctx = NoiseContext(grid, bfs_cache[goals[i]], bfs_cache)
            out.append(op(path, rng, ops, ctx, p))  # type: ignore[operator]
        else:
            out.append(op(path, rng, ops))  # type: ignore[operator]
    return out


def divergence_stats(
    true_paths: list[Path], noisy_paths: list[Path]
) -> tuple[float, float]:
    """(mean fraction of timesteps whose cell differs, mean relative length change).

    Works for any operator, including route-changing ones where
    `time_displacement` is undefined. Paths are compared over the longer of the
    two lengths, each padded with its final cell.
    """
    if not true_paths:
        return 0.0, 0.0

    fracs, ratios = [], []
    for t, n in zip(true_paths, noisy_paths):
        if not t:
            continue
        horizon = max(len(t), len(n))
        diff = sum(
            1
            for k in range(horizon)
            if (t[k] if k < len(t) else t[-1]) != (n[k] if k < len(n) else n[-1])
        )
        fracs.append(diff / horizon)
        ratios.append((len(n) - len(t)) / len(t))

    if not fracs:
        return 0.0, 0.0
    return sum(fracs) / len(fracs), sum(ratios) / len(ratios)
