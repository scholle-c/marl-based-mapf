# streamlit_app.py
from typing import List
from marl_path.model import TrainingStats
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import marl_path.visualization.settings as settings


def render_overview(training_stats: List[TrainingStats]) -> None:
    if not training_stats:
        st.info("No training-data selected")
        return

    st.title("Training Plots")
    selected = st.selectbox("Select displayed folder", settings.selected_folders)
    dist_table_type = st.selectbox("Select distance table type", ["Model", "LaCAM"])

    idx = settings.selected_folders.index(selected)
    selected_stats: TrainingStats = settings.train_stats[idx]

    if selected_stats.map_size is None:
        st.info(
            "No information available about the map size, so the distance tables cannot be displayed"
        )
        return

    map_width = selected_stats.map_size[1]
    map_heigh = selected_stats.map_size[0]

    if dist_table_type == "LaCAM":
        dist_tables = settings.dist_tables_lacam
    else:
        dist_tables = settings.dist_tables_model

    data = dist_tables[idx]

    n_blocks = data.shape[0] // map_heigh
    n_agents = data.shape[1] // map_width
    if n_agents > 1:
        selected_agent = st.slider("Agent", 0, n_agents - 1, 0)
    else:
        selected_agent = 0
    data = data[:, selected_agent * map_width : (selected_agent + 1) * map_width]

    stack = data.reshape(n_blocks, map_heigh, map_width)

    # Plotting Options:
    show_distances = st.checkbox("Show Distance Values", value=False)
    show_map = st.checkbox("Show Map", value=False, disabled=settings.map_mask is None)

    # Prüfe ob Pfad für den aktuellen Agenten existiert
    has_agent_path = (
        settings.agent_paths
        and len(settings.agent_paths) > 0
        and len(settings.agent_paths[0]) > selected_agent
    )
    show_agent_path = st.checkbox(
        "Show Agent Path", value=False, disabled=not has_agent_path
    )

    # Extrahiere Pfad für den ausgewählten Agenten
    agent_path_coords = []
    if has_agent_path and show_agent_path:
        if len(settings.agent_paths) > selected_agent:
            for t in range(len(settings.agent_paths[selected_agent])):
                coord = settings.agent_paths[selected_agent][t][0]  # [row, col]
                agent_path_coords.append(coord)

    # Wenn Show Map aktiviert ist und map_mask vorhanden, maskiere die Wände (map_mask=0)
    if show_map and settings.map_mask is not None:
        # Erstelle eine Kopie des Stacks und setze Wände auf NaN
        stack = stack.astype(float)
        mask = settings.map_mask == 0
        for i in range(n_blocks):
            stack[i][mask] = np.nan

    # Colorscale für Distanzen (Plasma hat guten Kontrast für sequentielle Daten)
    colorscale = "Plasma"

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=stack[0],
            colorscale=colorscale,
            text=stack[0] if show_distances else None,
            texttemplate="%{text}",
            textfont={"size": 12},
            showscale=True,
            hoverongaps=False,
        )
    )

    # Füge Agentenpfad hinzu wenn aktiviert
    if agent_path_coords:
        path_y = [coord[0] for coord in agent_path_coords]  # row = y
        path_x = [coord[1] for coord in agent_path_coords]  # col = x

        # Pfadlinie mit Markern
        fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=6, color="red", opacity=0.7),
                name="Path",
                hovertemplate="t=%{pointNumber}<br>x=%{x}, y=%{y}<extra></extra>",
            )
        )

        # Start markieren (grün)
        fig.add_trace(
            go.Scatter(
                x=[path_x[0]],
                y=[path_y[0]],
                mode="markers",
                marker=dict(
                    size=14,
                    color="lime",
                    symbol="circle",
                    line=dict(color="darkgreen", width=2),
                ),
                name="Start",
                hovertemplate="Start<br>x=%{x}, y=%{y}<extra></extra>",
            )
        )

        # Ziel markieren (blau)
        fig.add_trace(
            go.Scatter(
                x=[path_x[-1]],
                y=[path_y[-1]],
                mode="markers",
                marker=dict(
                    size=14,
                    color="cyan",
                    symbol="square",
                    line=dict(color="darkblue", width=2),
                ),
                name="Goal",
                hovertemplate="Goal<br>x=%{x}, y=%{y}<extra></extra>",
            )
        )

    fig.frames = [
        go.Frame(
            data=[
                go.Heatmap(
                    z=stack[i],
                    colorscale=colorscale,
                    text=stack[i] if show_distances else None,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    showscale=True,
                    hoverongaps=False,
                )
            ],
            name=str(i),
        )
        for i in range(n_blocks)
    ]

    steps = [
        dict(
            method="animate",
            args=[
                [str(i)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0},
                    "transition": {"duration": 0},
                },
            ],
            label=f"{i}",
        )
        for i in range(n_blocks)
    ]

    fig.update_layout(
        sliders=[dict(active=0, steps=steps, x=0.1, y=0, len=0.8)],
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="black",
        xaxis=dict(tickmode="linear", dtick=1),
        yaxis=dict(tickmode="linear", dtick=1),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(fig, width="stretch")


render_overview(settings.train_stats)
