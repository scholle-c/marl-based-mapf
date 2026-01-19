# streamlit_app.py
from typing import List
from marl_path.model import TrainingStats
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

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=stack[0],
            colorscale="Viridis",
            text=stack[0] if show_distances else None,
            texttemplate="%{text}",
            textfont={"size": 12},
            showscale=True,
        )
    )
    fig.frames = [
        go.Frame(
            data=[
                go.Heatmap(
                    z=stack[i],
                    colorscale="Viridis",
                    text=stack[i] if show_distances else None,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    showscale=True,
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
            label=str(i),
        )
        for i in range(n_blocks)
    ]

    fig.update_layout(
        sliders=[dict(active=0, steps=steps, x=0.1, y=0, len=0.8)],
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(fig, width="stretch")


render_overview(settings.train_stats)
