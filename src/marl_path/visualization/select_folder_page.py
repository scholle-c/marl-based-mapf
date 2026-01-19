import streamlit as st
import marl_path.visualization.settings as settings
from marl_path.visualization.plotting import load_stats, load_dist_tables


def go_up_click():
    settings.cwd = settings.cwd.parent


def go_down_click():
    name = st.session_state.get("selected_filename")
    if not name:
        return
    target = settings.cwd / name
    if target.is_dir():
        settings.cwd = target


st.title("MARL-Visualizer")
st.markdown("Select your root folder, that contains the training statistic folders:")

col1, col2, col3 = st.columns([5, 1, 1], gap="small")
with col1:
    subfolders = [p.name for p in settings.cwd.iterdir() if p.is_dir()]
    selected_filename = st.selectbox(
        "Folder",
        subfolders if subfolders else ["<no subfolders>"],
        key="selected_filename",
        label_visibility="collapsed",
    )
with col2:
    go_up_btn = st.button(
        "Up",
        icon="🔼",
        on_click=go_up_click,
        use_container_width=True,
    )
with col3:
    go_down_btn = st.button(
        "Open",
        icon="🔽",
        on_click=go_down_click,
        use_container_width=True,
    )

st.markdown("The following folders with training statistics have been found:")
training_dirs = [
    p
    for p in settings.cwd.iterdir()
    if p.is_dir() and (p / "training_stats.json").is_file()
]
training_dirs.sort(key=lambda p: p.name.lower())

selected_training = []
if training_dirs:
    for folder in training_dirs:
        key = f"training_folder_{folder.name}"
        if st.checkbox(folder.name, key=key, value=True):
            selected_training.append(folder)
else:
    st.info("No training folder found")

st.session_state["selected_training_folders"] = [str(p) for p in selected_training]

load_data_btn = st.button("Load selected data", disabled=not selected_training)

if load_data_btn:
    with st.spinner("Loading data..."):
        settings.train_stats = load_stats(selected_training)
        settings.selected_folders = selected_training.copy()
        settings.dist_tables_model, settings.dist_tables_lacam = load_dist_tables(
            selected_training
        )
    st.success("Done")
    st.switch_page(settings.OVERVIEW_PAGE)
