"""Model Explorer — load a checkpoint and inspect the predicted heuristic distance table."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

import marl_path.constants as consts
import marl_path.visualization.settings as settings
from marl_path.model.inference import load_model, predict_distance_table
from marl_path.shared.mapf_utils import get_grid, get_scenario, is_valid_coord


def _prefill_model_path() -> str:
    if settings.selected_folders:
        return str(
            Path(str(settings.selected_folders[0])) / consts.DEFAULT_FILENAME_TRAINED_MODEL
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
                marker=dict(symbol="square", size=10, color="orange", line=dict(width=1, color="white")),
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
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain", title="Row"),
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
    fig.add_trace(
        go.Scatter(
            x=[goal[1]],
            y=[goal[0]],
            mode="markers+text",
            marker=dict(symbol="star", size=16, color="red", line=dict(width=1, color="white")),
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
                symbol="circle", size=14, color="#00ff88", line=dict(width=1, color="white")
            ),
            text=["Start"],
            textposition="top center",
            name="Start",
        )
    )
    H, W = grid.shape
    cell_px = max(12, min(24, 600 // max(H, W)))
    fig.update_layout(
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain", title="Row"),
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

with st.expander("1  File paths", expanded="expl_grid" not in st.session_state):
    model_path: str = st.text_input(
        "Model checkpoint (.pt)",
        key="expl_model_path",
        placeholder="/path/to/trained_model.pt",
    )
    map_path: str = st.text_input(
        "Map file (.map)",
        key="expl_map_path",
        placeholder="/path/to/map.map",
    )
    scen_path: str = st.text_input(
        "Scenario file (.scen) — optional, provides other-agent positions",
        key="expl_scen_path",
        placeholder="/path/to/scenario.scen",
    )
    load_btn = st.button("Load files", disabled=not (model_path.strip() and map_path.strip()))

if load_btn:
    errors: list[str] = []
    with st.spinner("Loading files..."):
        try:
            m, ext = load_model(model_path.strip(), device=_auto_device())
            st.session_state["expl_model"] = m
            st.session_state["expl_extractor"] = ext
        except Exception as e:
            errors.append(f"Model — {e}")

        try:
            g = get_grid(map_path.strip())
            st.session_state["expl_grid"] = g
        except Exception as e:
            errors.append(f"Map — {e}")

        if scen_path.strip():
            try:
                sc_starts, sc_goals = get_scenario(scen_path.strip())
                st.session_state["expl_sc_starts"] = sc_starts
                st.session_state["expl_sc_goals"] = sc_goals
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

if grid is not None and model is not None:
    H, W = grid.shape
    st.caption(f"Map: {H} × {W} — {int(grid.sum())} traversable cells")

    with st.expander("2  Query", expanded=True):
        col_g, col_s = st.columns(2)
        with col_g:
            goal_row = st.number_input("Goal row", 0, H - 1, 0, key="expl_goal_row")
            goal_col = st.number_input("Goal col", 0, W - 1, 0, key="expl_goal_col")
        with col_s:
            start_row = st.number_input("Start row", 0, H - 1, 0, key="expl_start_row")
            start_col = st.number_input("Start col", 0, W - 1, 0, key="expl_start_col")

        other_positions: list[tuple[int, int]] = []
        if sc_starts is not None and sc_goals is not None:
            agent_labels = [
                f"Agent {i}  ({s[0]},{s[1]}) → ({g[0]},{g[1]})"
                for i, (s, g) in enumerate(zip(sc_starts, sc_goals))
            ]
            chosen = st.multiselect(
                "Other agents (from scenario)", agent_labels, key="expl_other_agents"
            )
            other_positions = [sc_starts[agent_labels.index(lbl)] for lbl in chosen]

        goal = (int(goal_row), int(goal_col))
        start = (int(start_row), int(start_col))
        goal_ok = is_valid_coord(grid, goal)
        start_ok = is_valid_coord(grid, start)
        if not goal_ok:
            st.warning("Goal is on a wall or out of bounds.")
        if not start_ok:
            st.warning("Start is on a wall or out of bounds.")

        st.plotly_chart(
            _make_map_preview(grid, start, goal, other_positions, start_ok, goal_ok),
            use_container_width=False,
        )

        predict_btn = st.button(
            "Predict heuristic table", disabled=not (goal_ok and start_ok)
        )

    if predict_btn:
        with st.spinner("Running model..."):
            pred = predict_distance_table(model, grid, start, goal, extractor, other_positions)
            st.session_state["expl_prediction"] = pred
            st.session_state["expl_query"] = (start, goal)

# ── 3. Heatmap result ────────────────────────────────────────────────────────

pred: np.ndarray | None = st.session_state.get("expl_prediction")
if pred is not None and grid is not None:
    last_start, last_goal = st.session_state.get("expl_query", ((0, 0), (0, 0)))
    st.subheader("Predicted heuristic table")
    fig = _make_heatmap(grid, pred, last_start, last_goal)
    st.plotly_chart(fig, use_container_width=False)

    free_vals = pred[grid]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min (free cells)", f"{free_vals.min():.2f}")
    c2.metric("Max (free cells)", f"{free_vals.max():.2f}")
    c3.metric("Mean (free cells)", f"{free_vals.mean():.2f}")
    at_start = pred[last_start] if is_valid_coord(grid, last_start) else None
    c4.metric("Value at start", f"{at_start:.2f}" if at_start is not None else "—")
