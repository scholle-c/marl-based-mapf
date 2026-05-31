from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import marl_path.visualization.settings as settings
import plotly.graph_objects as go
from marl_path.model.stats import TrainingStats
from marl_path.visualization.plotting import (
    align_runs,
    aggregate,
)

_CONFIG_FIELDS: List[tuple[str, str]] = [
    ("pipeline_mode", "Mode"),
    ("epochs", "Epochs"),
    ("batch_size", "Batch Size"),
    ("lr", "Learning Rate"),
    ("num_agents", "Agents"),
    ("map_file", "Map"),
    ("feature_extractor_type", "Extractor"),
    ("device", "Device"),
    ("seed", "Seed"),
    ("time_limit_ms", "Time Limit (ms)"),
    ("flg_star", "LaCAM*"),
]


def _fmt(key: str, val: Any) -> str:
    if val is None:
        return "—"
    if key == "map_file":
        return Path(str(val)).name
    return str(val)


def _configs_to_markdown(configs: List[Dict]) -> str:
    if len(configs) == 1:
        cfg = configs[0]
        lines = ["| Parameter | Value |", "|-----------|-------|"]
        for key, label in _CONFIG_FIELDS:
            lines.append(f"| {label} | {_fmt(key, cfg.get(key))} |")
    else:
        run_names = [Path(str(f)).name for f in settings.selected_folders]
        header = "| Parameter | " + " | ".join(run_names) + " |"
        sep = "|-----------|" + "|".join(["---"] * len(run_names)) + "|"
        lines = [header, sep]
        for key, label in _CONFIG_FIELDS:
            vals = " | ".join(_fmt(key, cfg.get(key)) for cfg in configs)
            lines.append(f"| {label} | {vals} |")
    return "\n".join(lines)


def render_config_info(configs: List[Dict]) -> None:
    if not configs:
        return
    with st.expander("Run configuration", expanded=True):
        if len(configs) == 1:
            cfg = configs[0]
            cols = st.columns(4)
            for i, (key, label) in enumerate(_CONFIG_FIELDS):
                cols[i % 4].markdown(f"**{label}**  \n{_fmt(key, cfg.get(key))}")
        else:
            rows = []
            for folder, cfg in zip(settings.selected_folders, configs):
                row: Dict[str, Any] = {"Run": Path(str(folder)).name}
                for key, label in _CONFIG_FIELDS:
                    row[label] = _fmt(key, cfg.get(key))
                rows.append(row)
            df = pd.DataFrame(rows).set_index("Run")

            # Highlight columns where values differ across runs
            def _highlight_diff(col):
                unique = col.dropna().unique()
                style = (
                    "background-color: #d97706; color: #000000"
                    if len(unique) > 1
                    else ""
                )
                return [style] * len(col)

            st.dataframe(
                df.style.apply(_highlight_diff, axis=0),
                use_container_width=True,
            )

        md = _configs_to_markdown(configs)
        with st.expander("Markdown", expanded=False):
            st.code(md, language="markdown")
            st.download_button(
                "Download .md",
                data=md,
                file_name="run_config.md",
                mime="text/markdown",
            )


def plot_single_series(
    title: str,
    y_label: str,
    values: List,
    color: str,
) -> go.Figure:
    epochs = list(range(1, len(values) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=values,
            mode="lines+markers",
            name=title,
            line=dict(color=color),
        )
    )
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title=y_label)
    return fig


def plot_multiple_series(
    title: str,
    y_label: str,
    runs: List[List],
    color: str,
    fill_color: str,
) -> go.Figure:
    aligned = align_runs(runs)
    mean_values, min_values, max_values = aggregate(aligned)
    epochs = list(range(1, len(mean_values) + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=max_values,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=min_values,
            fill="tonexty",
            fillcolor=fill_color,
            line=dict(width=0),
            name="Range",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=mean_values,
            mode="lines",
            name="Mean",
            line=dict(color=color, width=2),
        )
    )
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title=y_label)
    return fig


def add_range_band(
    fig: go.Figure,
    epochs: List[int],
    min_values: List,
    max_values: List,
    fill_color: str,
    name: str,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=max_values,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=min_values,
            fill="tonexty",
            fillcolor=fill_color,
            line=dict(width=0),
            name=name,
            hoverinfo="skip",
        )
    )


def plot_single_soc(stats: TrainingStats) -> go.Figure:
    socs = stats.mapf.socs if stats.mapf else []
    epochs = list(range(1, len(socs) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=socs,
            mode="lines+markers",
            name="SOC",
            line=dict(color="blue"),
        )
    )
    fig.update_layout(
        title="Sum of Costs (SOC) over Epochs",
        xaxis_title="Epoch",
        yaxis_title="SOC",
    )
    return fig


def plot_multiple_soc(training_stats: List[TrainingStats]) -> go.Figure:
    aligned_socs = align_runs(
        [stats.mapf.socs if stats.mapf else [] for stats in training_stats]
    )
    mean_socs, min_socs, max_socs = aggregate(aligned_socs)
    epochs = list(range(1, len(mean_socs) + 1))

    fig = go.Figure()
    add_range_band(
        fig,
        epochs,
        min_socs,
        max_socs,
        "rgba(30, 144, 255, 0.2)",
        "SOC range",
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=mean_socs,
            mode="lines",
            name="Mean SOC",
            line=dict(color="blue", width=2),
        )
    )

    fig.update_layout(
        title="Sum of Costs (SOC) over Epochs",
        xaxis_title="Epoch",
        yaxis_title="SOC",
    )
    return fig


_PLOTLY_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def plot_individual_series(
    title: str,
    y_label: str,
    runs: List[List],
    labels: List[str],
) -> go.Figure:
    fig = go.Figure()
    for i, (values, label) in enumerate(zip(runs, labels)):
        color = _PLOTLY_COLORS[i % len(_PLOTLY_COLORS)]
        epochs = list(range(1, len(values) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=values,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title=y_label)
    return fig


def render_overview(
    training_stats: List[TrainingStats], labels: List[str] | None = None
) -> None:
    if not training_stats:
        st.info("Keine Trainingsdaten geladen")
        return

    if len(training_stats) == 1:
        stats = training_stats[0]
        st.plotly_chart(plot_single_soc(stats), width="stretch")
        st.plotly_chart(
            plot_single_series(
                "Mean Loss over Epochs", "Mean Loss", stats._losses, color="orange"
            ),
            width="stretch",
        )
        elapsed_ms = stats.mapf.elapsed_times if stats.mapf else []
        if any(t is not None for t in elapsed_ms):
            elapsed_s = [t / 1000.0 if t is not None else None for t in elapsed_ms]
            st.plotly_chart(
                plot_single_series(
                    "Solve Time per Epoch", "Time (s)", elapsed_s, color="teal"
                ),
                width="stretch",
            )
        if stats.has_dist_table_differences:
            st.plotly_chart(
                plot_single_series(
                    "Distance Table Differences over Epochs",
                    "Mean Absolute Difference",
                    stats.dist_tables._diffs if stats.dist_tables is not None else [],
                    color="purple",
                ),
                width="stretch",
            )
        return

    run_labels = (
        labels
        if labels and len(labels) == len(training_stats)
        else [f"Run {i + 1}" for i in range(len(training_stats))]
    )
    losses = [stats._losses for stats in training_stats]
    all_elapsed = [
        [
            t / 1000.0 if t is not None else None
            for t in (s.mapf.elapsed_times if s.mapf else [])
        ]
        for s in training_stats
    ]
    all_socs = [stats.mapf.socs if stats.mapf else [] for stats in training_stats]

    display_mode = st.radio(
        "Display mode",
        ["Aggregated (mean ± range)", "Individual runs"],
        horizontal=True,
        key="overview_display_mode",
    )
    individual = display_mode == "Individual runs"

    if individual:
        st.plotly_chart(
            plot_individual_series(
                "Sum of Costs (SOC) over Epochs", "SOC", all_socs, run_labels
            ),
            width="stretch",
        )
        st.plotly_chart(
            plot_individual_series(
                "Mean Loss over Epochs", "Mean Loss", losses, run_labels
            ),
            width="stretch",
        )
        if any(any(t is not None for t in run) for run in all_elapsed):
            st.plotly_chart(
                plot_individual_series(
                    "Solve Time per Epoch", "Time (s)", all_elapsed, run_labels
                ),
                width="stretch",
            )
        if all(stats.has_dist_table_differences for stats in training_stats):
            dist_diffs = [
                stats.dist_tables._diffs
                for stats in training_stats
                if stats.dist_tables is not None
            ]
            st.plotly_chart(
                plot_individual_series(
                    "Distance Table Differences over Epochs",
                    "Mean Absolute Difference",
                    dist_diffs,
                    run_labels,
                ),
                width="stretch",
            )
    else:
        st.plotly_chart(plot_multiple_soc(training_stats), width="stretch")
        st.plotly_chart(
            plot_multiple_series(
                "Mean Loss over Epochs",
                "Mean Loss",
                losses,
                color="orange",
                fill_color="rgba(255, 165, 0, 0.2)",
            ),
            width="stretch",
        )
        if any(any(t is not None for t in run) for run in all_elapsed):
            st.plotly_chart(
                plot_multiple_series(
                    "Solve Time per Epoch",
                    "Time (s)",
                    all_elapsed,
                    color="teal",
                    fill_color="rgba(0, 128, 128, 0.2)",
                ),
                width="stretch",
            )
        if all(stats.has_dist_table_differences for stats in training_stats):
            dist_diffs = [
                stats.dist_tables._diffs
                for stats in training_stats
                if stats.dist_tables is not None
            ]
            st.plotly_chart(
                plot_multiple_series(
                    "Distance Table Differences over Epochs",
                    "Mean Absolute Difference",
                    dist_diffs,
                    color="purple",
                    fill_color="rgba(128, 0, 128, 0.2)",
                ),
                width="stretch",
            )
        else:
            st.info("Keine Distance-Table-Differences in den Daten gefunden")


def _coverage_heatmap(
    coverage: np.ndarray,
    title: str,
    map_mask: np.ndarray | None,
    as_pct: bool,
) -> go.Figure:
    z = coverage.astype(float)
    if as_pct:
        total = z.sum()
        z = z / total * 100 if total > 0 else z

    fig = go.Figure()
    if map_mask is not None:
        mask = map_mask.astype(bool)
        z = np.where(mask, z, np.nan)
        wall_z = np.where(mask, np.nan, 0.0)
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
    fmt = ".2f%%" if as_pct else "d"
    label = "%" if as_pct else "Count"
    fig.add_trace(
        go.Heatmap(
            z=z,
            colorscale="Blues",
            colorbar=dict(title=label),
            hovertemplate=f"row=%{{y}}, col=%{{x}}<br>value=%{{z:{fmt}}}<extra></extra>",
        )
    )
    H, W = coverage.shape
    cell_px = max(10, min(20, 500 // max(H, W)))
    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
        height=H * cell_px + 80,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def render_coverage(
    start_coverages: List,
    goal_coverages: List,
    labels: List[str],
    map_mask: np.ndarray | None,
) -> None:
    valid = [
        (sc, gc, lbl)
        for sc, gc, lbl in zip(start_coverages, goal_coverages, labels)
        if sc is not None and gc is not None
    ]
    if not valid:
        return
    with st.expander("Training Coverage", expanded=False):
        as_pct = st.toggle("Show as percentage", value=False, key="coverage_pct")
        for sc, gc, lbl in valid:
            if len(valid) > 1:
                st.markdown(f"**{lbl}**")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    _coverage_heatmap(sc, "Start positions", map_mask, as_pct),
                    use_container_width=True,
                )
            with col2:
                st.plotly_chart(
                    _coverage_heatmap(gc, "Goal positions", map_mask, as_pct),
                    use_container_width=True,
                )


def _build_summary_df(folders: List, stats_list: List[TrainingStats]) -> pd.DataFrame:
    rows = []
    for folder, stats in zip(folders, stats_list):
        name = Path(str(folder)).name
        socs = [s for s in (stats.mapf.socs if stats.mapf else []) if s is not None]
        losses = [loss for loss in stats._losses if loss is not None]
        elapsed_s = [
            t / 1000.0
            for t in (stats.mapf.elapsed_times if stats.mapf else [])
            if t is not None
        ]
        rows.append(
            {
                "Folder": name,
                "Epochs": stats._epoch_count,
                "Mean SOC": round(float(np.mean(socs)), 2) if socs else None,
                "Final SOC": socs[-1] if socs else None,
                "Mean Loss": round(float(np.mean(losses)), 4) if losses else None,
                "Final Loss": round(float(losses[-1]), 4) if losses else None,
                "Mean Solve Time (s)": round(float(np.mean(elapsed_s)), 3)
                if elapsed_s
                else None,
            }
        )
    return pd.DataFrame(rows)


st.title("Training Stats Overview")
st.subheader("Selected Folders:")
st.markdown(f"Root folder: {settings.cwd.name}")
if settings.selected_folders:
    st.markdown("\n".join(f"- {folder}" for folder in settings.selected_folders))
else:
    st.info("No child folders selected")
render_config_info(settings.run_configs)
_labels = [Path(str(f)).name for f in settings.selected_folders]
render_overview(settings.train_stats, labels=_labels)
render_coverage(
    settings.start_coverages, settings.goal_coverages, _labels, settings.map_mask
)

if settings.train_stats and settings.selected_folders:
    st.divider()
    st.subheader("Summary Statistics")
    df = _build_summary_df(settings.selected_folders, settings.train_stats)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Export as CSV",
        df.to_csv(index=False),
        file_name="summary.csv",
        mime="text/csv",
    )
