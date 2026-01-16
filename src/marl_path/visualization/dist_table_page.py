# streamlit_app.py
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import marl_path.constants as consts

BASE_DIR = "./output"
MAP_WIDTH = 4
MAP_HEIGHT = 6

st.title("Training Plots")
folders = sorted(
    d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))
)

selected = st.selectbox("Ergebnisordner", folders)
dist_table_type = st.selectbox("Distanztabellentyp", ["Model", "LaCAM"])

if dist_table_type == "LaCAM":
    filename = consts.DEFAULT_FILENAME_DIST_TABLE_LACAM
else:
    filename = consts.DEFAULT_FILENAME_DIST_TABLE_MODEL

csv_path = os.path.join(BASE_DIR, selected, filename)

data = np.loadtxt(csv_path, delimiter=",")
n_blocks = data.shape[0] // MAP_HEIGHT
n_agents = data.shape[1] // MAP_WIDTH
selected_agent = st.slider("Agent", 0, n_agents - 1, 0)
data = data[:, selected_agent * MAP_WIDTH : (selected_agent + 1) * MAP_WIDTH]

stack = data.reshape(n_blocks, MAP_HEIGHT, MAP_WIDTH)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=stack[0],
        colorscale="Viridis",
        text=stack[0],
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
                text=stack[i],
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
