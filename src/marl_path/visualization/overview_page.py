from typing import List, Tuple
import streamlit as st
import marl_path.visualization.settings as settings
import plotly.graph_objects as go
from marl_path.model.stats import TrainingStats


def align_runs(runs: List[List]) -> List[List]:
    """
    Truncate all runs to the shortest length so they can be combined safely.
    """
    runs = [list(run) for run in runs]
    if not runs:
        raise ValueError("No stats provided to align.")
    min_len = min(len(run) for run in runs)
    return [run[:min_len] for run in runs]


def aggregate(values: List[List]) -> Tuple[List, List, List]:
    """
    Calculate per-epoch mean, min and max across runs.
    """
    transposed = list(zip(*values))
    mean_values = [sum(v) / len(v) for v in transposed]
    min_values = [min(v) for v in transposed]
    max_values = [max(v) for v in transposed]
    return mean_values, min_values, max_values


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
    epochs = list(range(1, len(stats.socs) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=stats.socs,
            mode="lines+markers",
            name="SOC",
            line=dict(color="blue"),
        )
    )
    if stats.socs_model:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=stats.socs_model,
                mode="lines+markers",
                name="SOC with Model",
                line=dict(color="green"),
            )
        )
    if stats.socs_no_model:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=stats.socs_no_model,
                mode="lines+markers",
                name="SOC without Model",
                line=dict(color="red"),
            )
        )
    fig.update_layout(
        title="Sum of Costs (SOC) over Epochs",
        xaxis_title="Epoch",
        yaxis_title="SOC",
    )
    return fig


def plot_multiple_soc(training_stats: List[TrainingStats]) -> go.Figure:
    aligned_socs = align_runs([stats.socs for stats in training_stats])
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

    if all(stats.socs_model for stats in training_stats):
        aligned_soc_model = align_runs([stats.socs_model for stats in training_stats])
        mean_model, min_model, max_model = aggregate(aligned_soc_model)
        add_range_band(
            fig,
            epochs,
            min_model,
            max_model,
            "rgba(144, 238, 144, 0.2)",
            "SOC with Model range",
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=mean_model,
                mode="lines",
                name="Mean SOC with Model",
                line=dict(color="green", width=2),
            )
        )

    if all(stats.socs_no_model for stats in training_stats):
        aligned_soc_no_model = align_runs(
            [stats.socs_no_model for stats in training_stats]
        )
        mean_no_model, min_no_model, max_no_model = aggregate(aligned_soc_no_model)
        add_range_band(
            fig,
            epochs,
            min_no_model,
            max_no_model,
            "rgba(255, 160, 122, 0.2)",
            "SOC without Model range",
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=mean_no_model,
                mode="lines",
                name="Mean SOC without Model",
                line=dict(color="red", width=2),
            )
        )

    fig.update_layout(
        title="Sum of Costs (SOC) over Epochs",
        xaxis_title="Epoch",
        yaxis_title="SOC",
    )
    return fig


def render_overview(training_stats: List[TrainingStats]) -> None:
    if not training_stats:
        st.info("Keine Trainingsdaten geladen")
        return

    if len(training_stats) == 1:
        stats = training_stats[0]
        st.plotly_chart(
            plot_single_soc(stats),
            width="stretch",
        )
        st.plotly_chart(
            plot_single_series(
                "Mean Loss over Epochs",
                "Mean Loss",
                stats.training_loss,
                color="orange",
            ),
            width="stretch",
        )
        if stats.has_dist_table_differences:
            st.plotly_chart(
                plot_single_series(
                    "Distance Table Differences over Epochs",
                    "Mean Absolute Difference",
                    stats.dist_table_differences,
                    color="purple",
                ),
                width="stretch",
            )
        return

    losses = [stats.training_loss for stats in training_stats]
    st.plotly_chart(
        plot_multiple_soc(training_stats),
        width="stretch",
    )
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
    if all(stats.has_dist_table_differences for stats in training_stats):
        dist_diffs = [stats.dist_table_differences for stats in training_stats]
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


st.title("Training Stats Overview")
st.subheader("Selected Folders:")
st.markdown(f"Root folder: {settings.cwd.name}")
if settings.selected_folders:
    st.markdown("\n".join(f"- {folder}" for folder in settings.selected_folders))
else:
    st.info("No child folders selected")
render_overview(settings.train_stats)
