"""
Methods for visualizing the result of the reinforcement-learning training
"""

import argparse
import os
from typing import Iterable, List, Tuple
from marl_path.model.stats import TrainingStats
import matplotlib
import marl_path.constants as consts
import numpy as np

matplotlib.use(
    "TkAgg"  # Alternative "QTAgg"
)  # Remove if your running on windows or MacOS, only for linux systems with no display server
import matplotlib.pyplot as plt

SOC_PLOTTING_PARAMS = {
    "marker": "o",
    "label": "SOC",
    "label_multiple": "Mean SOC",
    "color": "blue",
}
SOC_MODEL_PLOTTING_PARAMS = {
    "marker": "x",
    "label": "SOC with Model",
    "label_multiple": "Mean SOC with Model",
    "color": "green",
}
SOC_NO_MODEL_PLOTTING_PARAMS = {
    "marker": "x",
    "label": "SOC without Model",
    "label_multiple": "Mean SOC without Model",
    "color": "red",
}
LOSSES_PLOTTING_PARAMS = {
    "marker": "o",
    "label": "Loss",
    "label_multiple": "Mean Loss",
    "color": "orange",
}
DIST_TABLE_DIFF_PLOTTING_PARAMS = {
    "marker": "o",
    "label": "Distance Table Differences",
    "label_multiple": "Mean Distance Table Differences",
    "color": "purple",
}


def run_plotting(args: argparse.Namespace) -> None:
    training_stats: List[TrainingStats] = load_stats(args.stats_file)
    if len(training_stats) == 1:
        plot_training_stats(training_stats[0])
    else:
        plot_multiple_training_stats(training_stats)


def load_stats(paths: List[str]) -> List[TrainingStats]:
    """
    Returns training statistics from a stats file or directory.
    Supports the new format (metrics.csv + config.json) and the legacy JSON format.
    """
    training_stats = []
    for path in paths:
        target_path = path
        if os.path.isdir(target_path):
            metrics = os.path.join(target_path, "metrics.csv")
            if os.path.isfile(metrics):
                training_stats.append(TrainingStats.load(target_path))
            else:
                json_files = _find_json_files_in_folder(
                    target_path, consts.DEFAULT_FILENAME_TRAINING_STATS
                )
                training_stats.extend(load_stats(json_files))
        elif not os.path.isfile(target_path):
            raise FileNotFoundError(f"Could not find stats file at {target_path}")
        else:
            training_stats.append(TrainingStats.load_from_json(target_path))
    return training_stats


def load_dist_tables(paths: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Looks in the provided folder or filepath for csv-files containing the history
    of the predicted distance tables. Files are identified via the filename in the
    constants.py.

    Args:
        paths (List[str]): One ore multiple paths to folders or csv files containing distance tables.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: First list contains the model-distance tables, the
        second list the distance tables from lacam.
    """
    dist_tables_lacam = _load_dist_tables(
        paths, consts.DEFAULT_FILENAME_DIST_TABLE_LACAM
    )
    dist_tables_model = _load_dist_tables(
        paths, consts.DEFAULT_FILENAME_DIST_TABLE_MODEL
    )
    return dist_tables_model, dist_tables_lacam


def _load_dist_tables(paths: List[str], filename: str) -> List[np.ndarray]:
    """
    Returns the distance table history from csv files.
    """
    dist_tables = []
    for path in paths:
        target_path = path
        if os.path.isdir(target_path):
            csv_files = _find_csv_files_in_folder(target_path, filename)
            dist_tables.extend(_load_dist_tables(csv_files, filename))
        elif not os.path.isfile(target_path):
            raise FileNotFoundError(f"Could not find dist-table file at {target_path}")
        else:
            dist_table = np.loadtxt(target_path, delimiter=",")
            dist_tables.append(dist_table)
    return dist_tables


def _find_json_files_in_folder(
    folder_path: str, filename: str | None = None
) -> List[str]:
    """
    _find_json_files_in_folder Looks for json files in a given folder and returns
    their filepath as a list.

    Args:
        folder_path (str): Path to the folder
        filename (str | None): If given, only json files with this name are returned.
    Returns:
        List[str]: A list of found json filepaths. Is empty if no json file was found.
    """
    json_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".json"):
                continue
            if filename is not None and file != filename:
                continue
            json_files.append(os.path.join(root, file))
    return json_files


def _find_csv_files_in_folder(
    folder_path: str, filename: str | None = None
) -> List[str]:
    """
    Looks for csv files in a given folder and returns their filepath as a list.

    Args:
        folder_path (str): Path to the folder
        filename (str | None): If given, only csv files with this name are returned.
    Returns:
        List[str]: A list of found csv filepaths. Is empty if no csv file was found.
    """
    csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".csv"):
                continue
            if filename is not None and file != filename:
                continue
            csv_files.append(os.path.join(root, file))
    return csv_files


def align_runs(runs: Iterable[List]) -> List[List]:
    """
    Truncate all runs to the shortest length so they can be combined safely.
    """
    runs = [list(run) for run in runs]
    if not runs:
        raise ValueError("No stats provided to align.")
    min_len = min(len(run) for run in runs)
    if any(len(run) != min_len for run in runs):
        print(f"Truncating all runs to {min_len} epochs to align lengths.")
    return [run[:min_len] for run in runs]


def aggregate(F: List[List]) -> Tuple[List, List, List]:
    """
    Calculate per-epoch mean, min and max across runs.
    """
    arr = np.array(
        [[np.nan if x is None else x for x in row] for row in F],
        dtype=float,
    ).T

    counts = np.sum(~np.isnan(arr), axis=1)
    sums = np.nansum(arr, axis=1)

    mean_values = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=float),
        where=counts != 0,
    ).tolist()
    with np.errstate(all="ignore"):
        min_values = np.nanmin(arr, axis=1).tolist()
        max_values = np.nanmax(arr, axis=1).tolist()
    return mean_values, min_values, max_values


def plot_training_stats(training_stats: TrainingStats) -> None:
    dist_table_differences = (
        training_stats.dist_tables._diffs
        if training_stats.dist_tables is not None and training_stats.dist_tables._diffs
        else None
    )
    _plot_training_stats(
        num_epochs=training_stats._epoch_count,
        socs=training_stats.mapf.socs if training_stats.mapf else [],
        losses=training_stats._losses,
        dist_table_differences=dist_table_differences,
    )


def plot_multiple_training_stats(training_stats_list: List[TrainingStats]) -> None:
    aligned_socs = align_runs(
        [stats.mapf.socs if stats.mapf else [] for stats in training_stats_list]
    )
    aligned_losses = align_runs([stats._losses for stats in training_stats_list])
    dist_table_differences = None
    if all(stats.has_dist_table_differences for stats in training_stats_list):
        dist_table_differences = align_runs(
            [
                stats.dist_tables._diffs
                for stats in training_stats_list
                if stats.dist_tables is not None
            ]
        )

    num_epochs = min(stats._epoch_count for stats in training_stats_list)
    _plot_multiple_training_stats(
        num_epochs=num_epochs,
        socs=aligned_socs,
        losses=aligned_losses,
        dist_table_differences=dist_table_differences,
    )


def _plot_training_stats(
    num_epochs: int,
    socs: List[int | None],
    losses: List[float | None],
    dist_table_differences: List | None = None,
    show: bool = True,
) -> None:
    """Plot training statistics of a single training run."""
    losses_cleaned: List[float] = [
        loss if loss is not None else np.nan for loss in losses
    ]
    socs_cleaned: List[float] = [soc if soc is not None else np.nan for soc in socs]

    plt.figure(figsize=(12, 5))
    epochs = list(range(1, num_epochs + 1))

    num_columns: int = 2
    used_dist_table_differences: bool = dist_table_differences is not None
    if used_dist_table_differences:
        num_columns = 3

    plt.subplot(1, num_columns, 1)
    plt.plot(
        epochs,
        socs_cleaned,
        color=SOC_PLOTTING_PARAMS["color"],
        marker=SOC_PLOTTING_PARAMS["marker"],
        label=SOC_PLOTTING_PARAMS["label"],
    )

    plt.title("Sum of Costs (SOC) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SOC")
    plt.legend()

    plt.subplot(1, num_columns, 2)
    plt.plot(
        epochs,
        losses_cleaned,
        marker=LOSSES_PLOTTING_PARAMS["marker"],
        color=LOSSES_PLOTTING_PARAMS["color"],
        label=LOSSES_PLOTTING_PARAMS["label"],
    )
    plt.title("Mean Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")
    plt.legend()

    if used_dist_table_differences:
        plt.subplot(1, num_columns, 3)
        plt.plot(
            epochs,
            dist_table_differences,
            marker=DIST_TABLE_DIFF_PLOTTING_PARAMS["marker"],
            color=DIST_TABLE_DIFF_PLOTTING_PARAMS["color"],
            label=DIST_TABLE_DIFF_PLOTTING_PARAMS["label"],
        )
        plt.title("Distance Table Differences over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Absolute Difference")
        plt.legend()

    plt.tight_layout()
    if show:
        plt.show()


def _plot_multiple_training_stats(
    num_epochs: int,
    socs: List[List[int]],
    losses: List[List[float]],
    dist_table_differences: List[List[float]] | None = None,
    show: bool = True,
) -> None:
    epochs = list(range(1, num_epochs + 1))

    num_columns: int = 2
    used_dist_table_differences: bool = dist_table_differences is not None
    if used_dist_table_differences:
        num_columns = 3

    plt.figure(figsize=(12, 5))
    plt.subplot(1, num_columns, 1)

    mean_socs, min_socs, max_socs = aggregate(socs)
    plt.fill_between(
        epochs, min_socs, max_socs, alpha=0.2, color="skyblue", label="SOC range"
    )
    plt.plot(
        epochs,
        mean_socs,
        color=SOC_PLOTTING_PARAMS["color"],
        linewidth=2,
        label=SOC_PLOTTING_PARAMS["label_multiple"],
    )

    plt.title("Sum of Costs (SOC) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SOC")
    plt.legend()

    plt.subplot(1, num_columns, 2)

    mean_losses, _, _ = aggregate(losses)
    for idx, loss in enumerate(losses):
        plt.plot(
            epochs,
            loss,
            color="orange",
            alpha=0.3,
            linewidth=1,
            label="Loss range" if idx == 0 else None,
        )
    plt.plot(
        epochs,
        mean_losses,
        color=LOSSES_PLOTTING_PARAMS["color"],
        linewidth=2,
        label=LOSSES_PLOTTING_PARAMS["label_multiple"],
    )

    plt.title("Mean Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")
    plt.legend()

    if used_dist_table_differences:
        plt.subplot(1, num_columns, 3)
        (
            mean_dist_table_differences,
            min_dist_table_differences,
            max_dist_table_differences,
        ) = aggregate(dist_table_differences)
        plt.plot(
            epochs,
            mean_dist_table_differences,
            marker=DIST_TABLE_DIFF_PLOTTING_PARAMS["marker"],
            color=DIST_TABLE_DIFF_PLOTTING_PARAMS["color"],
            label=DIST_TABLE_DIFF_PLOTTING_PARAMS["label_multiple"],
        )
        plt.fill_between(
            epochs,
            min_dist_table_differences,
            max_dist_table_differences,
            alpha=0.2,
            color="plum",
            # label="Distance Table Differences range",
        )
        plt.fill_between(
            epochs,
            min_dist_table_differences,
            max_dist_table_differences,
            alpha=0.2,
            color="plum",
            label="Distance Table Differences range",
        )
        plt.title("Distance Table Differences over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Absolute Difference")
        plt.legend()

    plt.tight_layout()
    if show:
        plt.show()
