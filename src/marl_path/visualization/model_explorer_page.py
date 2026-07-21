"""Model Explorer — load a checkpoint and inspect the predicted heuristic distance table."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

import marl_path.constants as consts
import marl_path.visualization.settings as settings
from marl_path.dataset import CachedInstance
from marl_path.model.inference import load_model, predict_distance_table
from marl_path.shared.mapf_utils import (
    BfsCache,
    get_grid,
    get_neighbors,
    get_scenario,
    is_valid_coord,
    resolve_portable_path,
)

Coord = tuple[int, int]


def _prefill_model_path() -> str:
    if settings.selected_folders:
        return str(
            Path(str(settings.selected_folders[0]))
            / consts.DEFAULT_FILENAME_TRAINED_MODEL
        )
    return ""


def _prefill_map_path() -> str:
    if settings.run_configs:
        return settings.run_configs[0].get("map_file", "")
    return ""


def _prefill_scen_path() -> str:
    if settings.run_configs:
        return settings.run_configs[0].get("scen_file", "")
    return ""


def _build_agent_labels(
    sc_starts, sc_goals, cbs_paths: dict[int, list[Coord]]
) -> list[str]:
    return [
        f"Agent {i}  ({s[0]},{s[1]}) → ({g[0]},{g[1]})"
        + ("  [CBS path]" if i in cbs_paths else "")
        for i, (s, g) in enumerate(zip(sc_starts, sc_goals))
    ]


def _on_fill_agent_changed() -> None:
    """on_change callback: push the selected agent's coords into the number-input keys."""
    label = st.session_state.get("expl_fill_agent", "")
    sc_starts = st.session_state.get("expl_sc_starts")
    sc_goals = st.session_state.get("expl_sc_goals")
    if not label or label == "— manual —" or sc_starts is None or sc_goals is None:
        return
    cbs_paths = st.session_state.get("expl_cbs_paths", {})
    agent_labels = _build_agent_labels(sc_starts, sc_goals, cbs_paths)
    if label not in agent_labels:
        return
    idx = agent_labels.index(label)
    s, g = sc_starts[idx], sc_goals[idx]
    st.session_state["expl_start_row"] = int(s[0])
    st.session_state["expl_start_col"] = int(s[1])
    st.session_state["expl_goal_row"] = int(g[0])
    st.session_state["expl_goal_col"] = int(g[1])


def _bfs_shortest_path(
    grid: np.ndarray, bfs_table: np.ndarray, start: Coord, goal: Coord
) -> list[Coord]:
    """Reconstruct a shortest path from start to goal by greedy descent on a BFS table."""
    if not is_valid_coord(grid, start) or not is_valid_coord(grid, goal):
        return []
    if bfs_table[start] >= bfs_table.size:
        return []  # unreachable
    path = [start]
    cur = start
    while cur != goal:
        neighbors = get_neighbors(grid, cur)
        if not neighbors:
            return []
        nxt = min(neighbors, key=lambda c: bfs_table[c])
        if bfs_table[nxt] >= bfs_table[cur]:
            return []  # stuck — shouldn't happen for a reachable goal
        path.append(nxt)
        cur = nxt
    return path


def _path_prediction_stats(
    pred: np.ndarray, grid: np.ndarray, path: list[Coord]
) -> dict[str, float] | None:
    """Compare the predicted table against a ground-truth path.

    The model predicts a delay/penalty in [0, 1]: 0 on the optimal path cell,
    1 elsewhere (see NonOptimalPenaltyDelay). So a good prediction has low
    values on `path` and high values off it.
    """
    if not path:
        return None
    on_vals = np.array([pred[c] for c in path], dtype=float)
    off_mask = grid.copy()
    for c in path:
        off_mask[c] = False
    off_vals = pred[off_mask]
    return {
        "path_len": float(len(path)),
        "on_mean": float(on_vals.mean()),
        "on_max": float(on_vals.max()),
        "off_mean": float(off_vals.mean()) if off_vals.size else float("nan"),
    }


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_map_preview(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    other_positions: list[tuple[int, int]],
    start_ok: bool,
    goal_ok: bool,
) -> go.Figure:
    H, W = grid.shape
    free_z = np.where(grid, 1.0, np.nan)
    wall_z = np.where(grid, np.nan, 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=wall_z,
            colorscale=[[0, "#888888"], [1, "#888888"]],
            showscale=False,
            hoverinfo="skip",
            zmin=0,
            zmax=1,
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=free_z,
            colorscale=[[0, "#e8e8e8"], [1, "#e8e8e8"]],
            showscale=False,
            hoverinfo="skip",
            zmin=0,
            zmax=1,
        )
    )
    if other_positions:
        fig.add_trace(
            go.Scatter(
                x=[p[1] for p in other_positions],
                y=[p[0] for p in other_positions],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=10,
                    color="orange",
                    line=dict(width=1, color="white"),
                ),
                name="Other agents",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[goal[1]],
            y=[goal[0]],
            mode="markers+text",
            marker=dict(
                symbol="star",
                size=16,
                color="red" if goal_ok else "#ffaaaa",
                line=dict(width=1, color="white"),
            ),
            text=["Goal"],
            textposition="top center",
            name="Goal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[start[1]],
            y=[start[0]],
            mode="markers+text",
            marker=dict(
                symbol="circle",
                size=14,
                color="#00ff88" if start_ok else "#aaffcc",
                line=dict(width=1, color="white"),
            ),
            text=["Start"],
            textposition="top center",
            name="Start",
        )
    )
    cell_px = max(12, min(24, 600 // max(H, W)))
    fig.update_layout(
        yaxis=dict(
            autorange="reversed", scaleanchor="x", constrain="domain", title="Row"
        ),
        xaxis=dict(constrain="domain", title="Col"),
        height=H * cell_px + 100,
        margin=dict(l=50, r=50, t=10, b=50),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def _make_heatmap(
    grid: np.ndarray,
    prediction: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    bfs_path: list[Coord] | None = None,
    cbs_path: list[Coord] | None = None,
) -> go.Figure:
    wall_z = np.where(grid, np.nan, 0.0)
    pred_z = np.where(grid, prediction.astype(float), np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=wall_z,
            colorscale=[[0, "#888888"], [1, "#888888"]],
            showscale=False,
            hoverinfo="skip",
            zmin=0,
            zmax=1,
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=pred_z,
            colorscale="Viridis_r",
            colorbar=dict(title="Heuristic"),
            hovertemplate="row=%{y}, col=%{x}<br>value=%{z:.2f}<extra></extra>",
        )
    )
    if bfs_path:
        fig.add_trace(
            go.Scatter(
                x=[c[1] for c in bfs_path],
                y=[c[0] for c in bfs_path],
                mode="lines+markers",
                line=dict(color="#00e5ff", width=3, dash="dash"),
                marker=dict(size=4, color="#00e5ff"),
                name="BFS shortest path",
                hovertemplate="t=%{pointNumber}<br>row=%{y}, col=%{x}<extra></extra>",
            )
        )
    if cbs_path:
        fig.add_trace(
            go.Scatter(
                x=[c[1] for c in cbs_path],
                y=[c[0] for c in cbs_path],
                mode="lines+markers",
                line=dict(color="#ff3d00", width=3),
                marker=dict(size=4, color="#ff3d00"),
                name="CBS dataset path",
                hovertemplate="t=%{pointNumber}<br>row=%{y}, col=%{x}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[goal[1]],
            y=[goal[0]],
            mode="markers+text",
            marker=dict(
                symbol="star", size=16, color="red", line=dict(width=1, color="white")
            ),
            text=["Goal"],
            textposition="top center",
            name="Goal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[start[1]],
            y=[start[0]],
            mode="markers+text",
            marker=dict(
                symbol="circle",
                size=14,
                color="#00ff88",
                line=dict(width=1, color="white"),
            ),
            text=["Start"],
            textposition="top center",
            name="Start",
        )
    )
    H, W = grid.shape
    cell_px = max(12, min(24, 600 // max(H, W)))
    fig.update_layout(
        yaxis=dict(
            autorange="reversed", scaleanchor="x", constrain="domain", title="Row"
        ),
        xaxis=dict(constrain="domain", title="Col"),
        height=H * cell_px + 100,
        margin=dict(l=50, r=50, t=30, b=50),
    )
    return fig


# ── page ─────────────────────────────────────────────────────────────────────

st.title("Model Explorer")
st.markdown(
    "Load a trained checkpoint and visualise the predicted heuristic distance table as a heatmap."
)

# ── 1. File paths ─────────────────────────────────────────────────────────────

# Initialise session-state keys with pre-fills only on first visit
for _key, _fn in [
    ("expl_model_path", _prefill_model_path),
    ("expl_map_path", _prefill_map_path),
    ("expl_scen_path", _prefill_scen_path),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _fn()
if "expl_dataset_path" not in st.session_state:
    st.session_state["expl_dataset_path"] = ""

with st.expander("1  File paths", expanded="expl_grid" not in st.session_state):
    model_path: str = st.text_input(
        "Model checkpoint (.pt)",
        key="expl_model_path",
        placeholder="/path/to/trained_model.pt",
    )
    map_path: str = st.text_input(
        "Map file (.map) — optional if a dataset instance is given below",
        key="expl_map_path",
        placeholder="/path/to/map.map",
    )
    scen_path: str = st.text_input(
        "Scenario file (.scen) — optional, provides other-agent positions",
        key="expl_scen_path",
        placeholder="/path/to/scenario.scen",
    )
    dataset_path: str = st.text_input(
        "CBS dataset instance (.npz) — optional, provides ground-truth CBS-optimal paths",
        key="expl_dataset_path",
        placeholder="/path/to/instance.npz",
    )
    load_btn = st.button(
        "Load files",
        disabled=not (
            model_path.strip() and (map_path.strip() or dataset_path.strip())
        ),
    )

if load_btn:
    errors: list[str] = []
    with st.spinner("Loading files..."):
        try:
            m, ext = load_model(model_path.strip(), device=_auto_device())
            st.session_state["expl_model"] = m
            st.session_state["expl_extractor"] = ext
        except Exception as e:
            errors.append(f"Model — {e}")

        dataset_inst: CachedInstance | None = None
        if dataset_path.strip():
            try:
                dataset_inst = CachedInstance.load(Path(dataset_path.strip()))
                st.session_state["expl_dataset"] = dataset_inst
                st.session_state["expl_cbs_paths"] = {
                    idx: path
                    for idx, path in zip(dataset_inst.agent_indices, dataset_inst.paths)
                    if path
                }
            except Exception as e:
                errors.append(f"Dataset — {e}")
        else:
            st.session_state.pop("expl_dataset", None)
            st.session_state.pop("expl_cbs_paths", None)

        effective_map_path = map_path.strip() or (
            str(resolve_portable_path(dataset_inst.map_file)) if dataset_inst else ""
        )
        effective_scen_path = scen_path.strip() or (
            str(resolve_portable_path(dataset_inst.scen_file)) if dataset_inst else ""
        )

        try:
            g = get_grid(effective_map_path)
            st.session_state["expl_grid"] = g
            st.session_state["expl_bfs_cache"] = BfsCache(g)
            st.session_state["expl_effective_map_path"] = effective_map_path
        except Exception as e:
            errors.append(f"Map — {e}")

        if effective_scen_path:
            try:
                sc_starts, sc_goals = get_scenario(effective_scen_path)
                st.session_state["expl_sc_starts"] = sc_starts
                st.session_state["expl_sc_goals"] = sc_goals
                st.session_state["expl_effective_scen_path"] = effective_scen_path
            except Exception as e:
                errors.append(f"Scenario — {e}")
        else:
            st.session_state.pop("expl_sc_starts", None)
            st.session_state.pop("expl_sc_goals", None)

        st.session_state.pop("expl_prediction", None)

    for err in errors:
        st.error(err)
    if not errors:
        st.success("Loaded successfully.")
        st.rerun()

# ── 2. Query ─────────────────────────────────────────────────────────────────

grid: np.ndarray | None = st.session_state.get("expl_grid")
model = st.session_state.get("expl_model")
extractor = st.session_state.get("expl_extractor")
sc_starts = st.session_state.get("expl_sc_starts")
sc_goals = st.session_state.get("expl_sc_goals")
bfs_cache: BfsCache | None = st.session_state.get("expl_bfs_cache")
cbs_paths: dict[int, list[Coord]] = st.session_state.get("expl_cbs_paths", {})

if grid is not None and model is not None:
    H, W = grid.shape
    st.caption(f"Map: {H} × {W} — {int(grid.sum())} traversable cells")
    if cbs_paths:
        st.caption(
            f"CBS dataset loaded — {len(cbs_paths)} agent(s) have a ground-truth path."
        )

    with st.expander("2  Query", expanded=True):
        agent_labels: list[str] = []
        if sc_starts is not None and sc_goals is not None:
            agent_labels = _build_agent_labels(sc_starts, sc_goals, cbs_paths)
            st.selectbox(
                "Fill start & goal from agent",
                ["— manual —"] + agent_labels,
                key="expl_fill_agent",
                on_change=_on_fill_agent_changed,
            )

        col_g, col_s = st.columns(2)
        with col_g:
            goal_row = st.number_input("Goal row", 0, H - 1, 0, key="expl_goal_row")
            goal_col = st.number_input("Goal col", 0, W - 1, 0, key="expl_goal_col")
        with col_s:
            start_row = st.number_input("Start row", 0, H - 1, 0, key="expl_start_row")
            start_col = st.number_input("Start col", 0, W - 1, 0, key="expl_start_col")

        other_starts: list[tuple[int, int]] = []
        other_goals: list[tuple[int, int]] = []
        if agent_labels:
            assert sc_starts is not None
            assert sc_goals is not None
            chosen = st.multiselect(
                "Other agents (from scenario)", agent_labels, key="expl_other_agents"
            )
            chosen_idx = [agent_labels.index(lbl) for lbl in chosen]
            other_starts = [sc_starts[i] for i in chosen_idx]
            other_goals = [sc_goals[i] for i in chosen_idx]

        goal = (int(goal_row), int(goal_col))
        start = (int(start_row), int(start_col))
        goal_ok = is_valid_coord(grid, goal)
        start_ok = is_valid_coord(grid, start)
        if not goal_ok:
            st.warning("Goal is on a wall or out of bounds.")
        if not start_ok:
            st.warning("Start is on a wall or out of bounds.")

        st.plotly_chart(
            _make_map_preview(grid, start, goal, other_starts, start_ok, goal_ok),
            use_container_width=False,
        )

        predict_btn = st.button(
            "Predict heuristic table", disabled=not (goal_ok and start_ok)
        )

    if predict_btn:
        with st.spinner("Running model..."):
            pred = predict_distance_table(
                model,
                grid,
                start,
                goal,
                extractor,
                other_agents=other_goals,
                other_starts=other_starts,
                bfs_cache=bfs_cache,
            )
            st.session_state["expl_prediction"] = pred
            st.session_state["expl_query"] = (start, goal)

            # Ground-truth paths for comparison: BFS shortest path always
            # available; CBS path only if the selected agent has one and it
            # matches the current start/goal.
            bfs_path: list[Coord] = []
            if bfs_cache is not None:
                bfs_path = _bfs_shortest_path(grid, bfs_cache[goal], start, goal)

            cbs_path: list[Coord] = []
            fill_label = st.session_state.get("expl_fill_agent", "")
            if agent_labels and fill_label in agent_labels:
                agent_idx = agent_labels.index(fill_label)
                candidate = cbs_paths.get(agent_idx, [])
                if candidate and candidate[0] == start and candidate[-1] == goal:
                    cbs_path = candidate

            st.session_state["expl_ground_truth"] = {
                "bfs_path": bfs_path,
                "cbs_path": cbs_path,
            }

# ── 3. Heatmap result ────────────────────────────────────────────────────────

pred: np.ndarray | None = st.session_state.get("expl_prediction")
if pred is not None and grid is not None:
    last_start, last_goal = st.session_state.get("expl_query", ((0, 0), (0, 0)))
    ground_truth = st.session_state.get("expl_ground_truth", {})
    bfs_path = ground_truth.get("bfs_path", [])
    cbs_path = ground_truth.get("cbs_path", [])

    st.subheader("Predicted heuristic table")
    st.caption(
        "The model predicts a delay/penalty in [0, 1]: values near 0 mean "
        "'on the optimal path', values near 1 mean 'far from it'."
    )
    fig = _make_heatmap(grid, pred, last_start, last_goal, bfs_path, cbs_path)
    st.plotly_chart(fig, use_container_width=False)

    free_vals = pred[grid]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min (free cells)", f"{free_vals.min():.2f}")
    c2.metric("Max (free cells)", f"{free_vals.max():.2f}")
    c3.metric("Mean (free cells)", f"{free_vals.mean():.2f}")
    at_start = pred[last_start] if is_valid_coord(grid, last_start) else None
    c4.metric("Value at start", f"{at_start:.2f}" if at_start is not None else "—")

    # ── Ground-truth comparison ─────────────────────────────────────────────
    if bfs_path or cbs_path:
        st.subheader("Comparison against the optimal path")
        for label, path in [
            ("BFS shortest path", bfs_path),
            ("CBS dataset path", cbs_path),
        ]:
            stats = _path_prediction_stats(pred, grid, path)
            if stats is None:
                continue
            st.markdown(f"**{label}** ({int(stats['path_len'])} steps)")
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Mean value on path",
                f"{stats['on_mean']:.3f}",
                help="Lower is better — 0 means the model correctly scores this cell as optimal.",
            )
            m2.metric(
                "Max value on path",
                f"{stats['on_max']:.3f}",
                help="Worst single cell along the path.",
            )
            m3.metric(
                "Mean value off path",
                f"{stats['off_mean']:.3f}",
                help="Higher is better — ideally close to 1.",
            )

        reference_path = cbs_path or bfs_path
        if reference_path:
            path_vals = [float(pred[c]) for c in reference_path]
            fig_line = go.Figure()
            fig_line.add_trace(
                go.Scatter(
                    x=list(range(len(path_vals))),
                    y=path_vals,
                    mode="lines+markers",
                    name="Predicted value along path",
                    line=dict(color="#ff3d00" if cbs_path else "#00e5ff"),
                )
            )
            fig_line.update_layout(
                title="Predicted value at each step of the ground-truth path (0 = optimal)",
                xaxis_title="Step along path",
                yaxis_title="Predicted value",
                height=300,
                margin=dict(l=50, r=50, t=40, b=40),
            )
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info(
            "No ground-truth path available for this start/goal — "
            "either it is unreachable, or (for a CBS path) the selected agent "
            "has no cached CBS solution."
        )
