from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import marl_path.visualization.settings as settings
from marl_path.model.stats import TrainingStats
from marl_path.visualization.plotting import aggregate, align_runs, load_stats

_COLORS = [
    ("#1f77b4", "rgba(31,119,180,0.2)"),
    ("#ff7f0e", "rgba(255,127,14,0.2)"),
    ("#2ca02c", "rgba(44,160,44,0.2)"),
    ("#d62728", "rgba(214,39,40,0.2)"),
    ("#9467bd", "rgba(148,103,189,0.2)"),
    ("#8c564b", "rgba(140,86,75,0.2)"),
    ("#e377c2", "rgba(227,119,194,0.2)"),
    ("#7f7f7f", "rgba(127,127,127,0.2)"),
]


# ---------------------------------------------------------------------------
# Folder navigation state (local to this page via session_state)
# ---------------------------------------------------------------------------

if "cmp_cwd" not in st.session_state:
    st.session_state.cmp_cwd = settings.cwd


def _go_up() -> None:
    st.session_state.cmp_cwd = st.session_state.cmp_cwd.parent


def _go_down() -> None:
    name = st.session_state.get("cmp_selected_filename")
    if name:
        target = st.session_state.cmp_cwd / name
        if target.is_dir():
            st.session_state.cmp_cwd = target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_variant_folder(p: Path) -> bool:
    """True if p contains at least one subfolder that has a metrics.csv."""
    return p.is_dir() and any(
        sub.is_dir() and (sub / "metrics.csv").is_file() for sub in p.iterdir()
    )


def _plot_metric_comparison(
    title: str,
    y_label: str,
    series: Dict[str, List[List]],
) -> go.Figure:
    fig = go.Figure()
    for i, (name, runs) in enumerate(series.items()):
        color, fill_color = _COLORS[i % len(_COLORS)]
        valid_runs = [r for r in runs if r]
        if not valid_runs:
            continue
        try:
            aligned = align_runs(valid_runs)
        except ValueError:
            continue
        mean_vals, min_vals, max_vals = aggregate(aligned)
        epochs = list(range(1, len(mean_vals) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=max_vals,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=min_vals,
                fill="tonexty",
                fillcolor=fill_color,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=mean_vals,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title=y_label)
    return fig


def _stats_for_variant(stats_list: List[TrainingStats]) -> dict:
    all_socs = [
        s
        for st_ in stats_list
        for s in (st_.mapf.socs if st_.mapf else [])
        if s is not None
    ]
    all_losses = [
        loss for st_ in stats_list for loss in st_._losses if loss is not None
    ]
    final_socs = [
        st_.mapf.socs[-1]
        for st_ in stats_list
        if st_.mapf and st_.mapf.socs and st_.mapf.socs[-1] is not None
    ]
    final_losses = [
        st_._losses[-1]
        for st_ in stats_list
        if st_._losses and st_._losses[-1] is not None
    ]
    all_elapsed_s = [
        t / 1000.0
        for st_ in stats_list
        for t in (st_.mapf.elapsed_times if st_.mapf else [])
        if t is not None
    ]
    return {
        "n_runs": len(stats_list),
        "min_epochs": min(s._epoch_count for s in stats_list),
        "all_socs": all_socs,
        "all_losses": all_losses,
        "final_socs": final_socs,
        "final_losses": final_losses,
        "all_elapsed_s": all_elapsed_s,
        "training_mode": stats_list[0].training_mode if stats_list else "",
    }


def _build_comparison_df(data: Dict[str, List[TrainingStats]]) -> pd.DataFrame:
    rows = []
    for name, stats_list in data.items():
        v = _stats_for_variant(stats_list)
        rows.append(
            {
                "Variant": name,
                "Runs": v["n_runs"],
                "Epochs (min)": v["min_epochs"],
                "Mean SOC": round(float(np.mean(v["all_socs"])), 2)
                if v["all_socs"]
                else None,
                "Mean Final SOC": round(float(np.mean(v["final_socs"])), 2)
                if v["final_socs"]
                else None,
                "Std Final SOC": round(float(np.std(v["final_socs"])), 2)
                if len(v["final_socs"]) > 1
                else None,
                "Mean Loss": round(float(np.mean(v["all_losses"])), 4)
                if v["all_losses"]
                else None,
                "Mean Final Loss": round(float(np.mean(v["final_losses"])), 4)
                if v["final_losses"]
                else None,
                "Std Final Loss": round(float(np.std(v["final_losses"])), 4)
                if len(v["final_losses"]) > 1
                else None,
                "Mean Solve Time (s)": round(float(np.mean(v["all_elapsed_s"])), 3)
                if v["all_elapsed_s"]
                else None,
            }
        )
    return pd.DataFrame(rows)


def _fmt(value, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _md_table(headers: List[str], rows: List[List]) -> str:
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    header_row = "| " + " | ".join(headers) + " |"
    data_rows = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([header_row, sep] + data_rows)


def _generate_report_md(
    data: Dict[str, List[TrainingStats]],
    run_names: Dict[str, List[str]],
    root_dir: Path,
) -> str:
    today = date.today().isoformat()
    lines: List[str] = []

    # ---- header ----
    lines += [
        "# Benchmark Comparison Report",
        "",
        f"**Generated:** {today}  ",
        f"**Root directory:** `{root_dir}`  ",
        f"**Variants compared:** {len(data)}  ",
        "",
        "---",
        "",
    ]

    # ---- summary table ----
    lines += ["## Summary", ""]
    headers = [
        "Variant",
        "Runs",
        "Epochs",
        "Mean SOC",
        "Mean Final SOC",
        "Std Final SOC",
        "Mean Loss",
        "Mean Final Loss",
        "Std Final Loss",
        "Mean Solve Time (s)",
    ]
    table_rows = []
    for name, stats_list in data.items():
        v = _stats_for_variant(stats_list)
        table_rows.append(
            [
                name,
                v["n_runs"],
                v["min_epochs"],
                _fmt(np.mean(v["all_socs"]) if v["all_socs"] else None),
                _fmt(np.mean(v["final_socs"]) if v["final_socs"] else None),
                _fmt(np.std(v["final_socs"]) if len(v["final_socs"]) > 1 else None),
                _fmt(np.mean(v["all_losses"]) if v["all_losses"] else None, 4),
                _fmt(np.mean(v["final_losses"]) if v["final_losses"] else None, 4),
                _fmt(
                    np.std(v["final_losses"]) if len(v["final_losses"]) > 1 else None, 4
                ),
                _fmt(np.mean(v["all_elapsed_s"]) if v["all_elapsed_s"] else None, 3),
            ]
        )
    lines += [_md_table(headers, table_rows), "", "---", ""]

    # ---- per-variant details ----
    lines += ["## Per-Variant Details", ""]
    for name, stats_list in data.items():
        v = _stats_for_variant(stats_list)
        lines += [f"### {name}", ""]

        if v["training_mode"]:
            lines += [f"- **Training mode:** `{v['training_mode']}`"]
        lines += [
            f"- **Runs:** {v['n_runs']}",
            f"- **Epochs per run (min):** {v['min_epochs']}",
        ]

        # config info from first run
        first = stats_list[0]
        if first.mapf:
            if first.mapf.num_agents is not None:
                lines.append(f"- **Agents:** {first.mapf.num_agents}")
            if first.mapf.map_size is not None:
                lines.append(
                    f"- **Map size:** {first.mapf.map_size[0]} × {first.mapf.map_size[1]}"
                )
        if first.used_device:
            lines.append(f"- **Device:** {first.used_device}")
        lines.append("")

        # per-run table
        run_labels = run_names.get(
            name, [f"run{i + 1}" for i in range(len(stats_list))]
        )
        run_headers = [
            "Run",
            "Epochs",
            "Final SOC",
            "Final Loss",
            "Mean Solve Time (s)",
        ]
        run_rows = []
        for label, st_ in zip(run_labels, stats_list):
            socs = [s for s in (st_.mapf.socs if st_.mapf else []) if s is not None]
            losses = [loss for loss in st_._losses if loss is not None]
            elapsed_s = [
                t / 1000.0
                for t in (st_.mapf.elapsed_times if st_.mapf else [])
                if t is not None
            ]
            run_rows.append(
                [
                    label,
                    st_._epoch_count,
                    _fmt(socs[-1]) if socs else "—",
                    _fmt(losses[-1], 4) if losses else "—",
                    _fmt(np.mean(elapsed_s), 3) if elapsed_s else "—",
                ]
            )
        lines += [_md_table(run_headers, run_rows), ""]

        # aggregated metrics block
        lines += ["**Aggregated across all runs:**", ""]
        if v["final_socs"]:
            lines.append(
                f"- SOC (final epoch): mean = {np.mean(v['final_socs']):.2f}"
                + (
                    f", std = {np.std(v['final_socs']):.2f}"
                    if len(v["final_socs"]) > 1
                    else ""
                )
            )
        if v["final_losses"]:
            lines.append(
                f"- Loss (final epoch): mean = {np.mean(v['final_losses']):.4f}"
                + (
                    f", std = {np.std(v['final_losses']):.4f}"
                    if len(v["final_losses"]) > 1
                    else ""
                )
            )
        if v["all_elapsed_s"]:
            lines.append(f"- Mean solve time: {np.mean(v['all_elapsed_s']):.3f} s")
        lines += ["", "---", ""]

    # ---- notes ----
    lines += [
        "## Notes",
        "",
        "- **SOC** (Sum of Costs): total number of timesteps across all agents. Lower is better.",
        "- **Loss**: VDN training loss. Lower is better.",
        "- **Solve Time**: time LaCAM spent searching for a solution per epoch (milliseconds converted to seconds).",
        "- Bands in charts represent min/max across runs; lines represent the mean.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("Benchmark Comparison")
st.markdown(
    "Navigate to a folder containing your **variant folders** "
    "(each variant folder should contain run sub-folders with `metrics.csv`)."
)

cwd: Path = st.session_state.cmp_cwd
col1, col2, col3 = st.columns([5, 1, 1], gap="small")
with col1:
    subfolders = [p.name for p in cwd.iterdir() if p.is_dir()]
    st.selectbox(
        "Folder",
        subfolders if subfolders else ["<no subfolders>"],
        key="cmp_selected_filename",
        label_visibility="collapsed",
    )
with col2:
    st.button("Up", icon="🔼", on_click=_go_up, use_container_width=True)
with col3:
    st.button("Open", icon="🔽", on_click=_go_down, use_container_width=True)

st.caption(f"Current directory: {cwd}")

# Detect variant folders in the current directory
variant_dirs = sorted(
    [p for p in cwd.iterdir() if _is_variant_folder(p)],
    key=lambda p: p.name.lower(),
)

st.markdown("**Detected variant folders:**")
selected_variant_dirs: Dict[str, Path] = {}
if variant_dirs:
    for vdir in variant_dirs:
        run_count = sum(
            1
            for sub in vdir.iterdir()
            if sub.is_dir() and (sub / "metrics.csv").is_file()
        )
        if st.checkbox(
            f"{vdir.name}  ({run_count} run{'s' if run_count != 1 else ''})",
            key=f"cmp_cb_{vdir.name}",
            value=True,
        ):
            selected_variant_dirs[vdir.name] = vdir
else:
    st.info(
        "No variant folders found here. A variant folder must contain at least one "
        "sub-folder with `metrics.csv`."
    )

load_btn = st.button("Load & Compare", disabled=not selected_variant_dirs)

if load_btn:
    with st.spinner("Loading runs…"):
        new_data: Dict[str, List[TrainingStats]] = {}
        new_run_names: Dict[str, List[str]] = {}
        for vname, vdir in selected_variant_dirs.items():
            run_dirs = sorted(
                [
                    sub
                    for sub in vdir.iterdir()
                    if sub.is_dir() and (sub / "metrics.csv").is_file()
                ],
                key=lambda p: p.name,
            )
            new_data[vname] = load_stats([str(d) for d in run_dirs])
            new_run_names[vname] = [d.name for d in run_dirs]
        settings.comparison_variants = new_data
        settings.comparison_variant_run_names = new_run_names
    st.success(
        f"Loaded {sum(len(v) for v in settings.comparison_variants.values())} runs "
        f"across {len(settings.comparison_variants)} variants."
    )

if not settings.comparison_variants:
    st.stop()

# ---------------------------------------------------------------------------
# Comparison charts
# ---------------------------------------------------------------------------

st.divider()
st.subheader("SOC over Epochs")
st.plotly_chart(
    _plot_metric_comparison(
        "Mean SOC per Epoch (with min/max band)",
        "SOC",
        {
            name: [s.mapf.socs if s.mapf else [] for s in stats_list]
            for name, stats_list in settings.comparison_variants.items()
        },
    ),
    use_container_width=True,
)

st.subheader("Loss over Epochs")
st.plotly_chart(
    _plot_metric_comparison(
        "Mean Loss per Epoch (with min/max band)",
        "Mean Loss",
        {
            name: [s._losses for s in stats_list]
            for name, stats_list in settings.comparison_variants.items()
        },
    ),
    use_container_width=True,
)

elapsed_series = {
    name: [
        [
            t / 1000.0 if t is not None else None
            for t in (s.mapf.elapsed_times if s.mapf else [])
        ]
        for s in stats_list
    ]
    for name, stats_list in settings.comparison_variants.items()
}
if any(
    any(any(t is not None for t in run) for run in runs)
    for runs in elapsed_series.values()
):
    st.subheader("Solve Time per Epoch")
    st.plotly_chart(
        _plot_metric_comparison(
            "Mean Solve Time per Epoch (with min/max band)",
            "Time (s)",
            elapsed_series,
        ),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Summary Table")
df = _build_comparison_df(settings.comparison_variants)
st.dataframe(df, use_container_width=True)

col_csv, col_md = st.columns(2)
with col_csv:
    st.download_button(
        "Export Summary CSV",
        df.to_csv(index=False),
        file_name="comparison_summary.csv",
        mime="text/csv",
    )
with col_md:
    report_md = _generate_report_md(
        settings.comparison_variants,
        settings.comparison_variant_run_names,
        st.session_state.cmp_cwd,
    )
    st.download_button(
        "Create Report (Markdown)",
        report_md,
        file_name="benchmark_report.md",
        mime="text/markdown",
    )
