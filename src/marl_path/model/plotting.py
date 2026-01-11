"""
Methods for visualizing the result of the reinforcement-learning training
"""

import argparse
import os
from typing import Iterable, List, Tuple
from marl_path.model.stats import TrainingStats
import matplotlib

matplotlib.use(
    "Qt5Agg"
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot training statistics for distance table CNN model."
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Path(s) to JSON stats files or folders containing training_stats.json. "
            "Provide multiple to visualize distribution."
        ),
    )
    args = parser.parse_args()
    training_stats: List[TrainingStats] = _load_stats(args.stats_file)
    if len(training_stats) == 1:
        plot_training_stats(training_stats[0])
    else:
        plot_multiple_training_stats(training_stats)


def _load_stats(paths: List[str]) -> List[TrainingStats]:
    """
    Returns training statistics from a stats JSON file.
    """
    training_stats = []
    for path in paths:
        target_path = path
        if os.path.isdir(target_path):
            json_files = _find_json_files_in_folder(target_path)
            training_stats.extend(_load_stats(json_files))
        elif not os.path.isfile(target_path):
            raise FileNotFoundError(f"Could not find stats file at {target_path}")
        else:
            stats = TrainingStats.load_from_json(target_path)
            training_stats.append(stats)
    return training_stats


def _find_json_files_in_folder(folder_path: str) -> List[str]:
    """
    _find_json_files_in_folder Looks for json files in a given folder and returns
    their filepath as a list.

    Args:
        folder_path (str): Path to the folder

    Returns:
        List[str]: A list of found json filepaths. Is empty if no json file was found.
    """
    json_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def _align_runs(runs: Iterable[List]) -> List[List]:
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


def _aggregate(values: List[List]) -> Tuple[List, List, List]:
    """
    Calculate per-epoch mean, min and max across runs.
    """
    transposed = list(zip(*values))
    mean_values = [sum(v) / len(v) for v in transposed]
    min_values = [min(v) for v in transposed]
    max_values = [max(v) for v in transposed]
    return mean_values, min_values, max_values


def plot_training_stats(training_stats: TrainingStats) -> None:
    socs_model = training_stats.socs_model if training_stats.used_best_mode else None
    socs_no_model = (
        training_stats.socs_no_model if training_stats.used_best_mode else None
    )
    dist_table_differences = (
        training_stats.dist_table_differences
        if training_stats.has_dist_table_differences
        else None
    )
    _plot_training_stats(
        num_epochs=training_stats.epochs,
        socs=training_stats.socs,
        losses=training_stats.training_loss,
        socs_model=socs_model,
        socs_no_model=socs_no_model,
        dist_table_differences=dist_table_differences,
    )


def plot_multiple_training_stats(training_stats_list: List[TrainingStats]) -> None:
    aligned_socs = _align_runs([stats.socs for stats in training_stats_list])
    aligned_losses = _align_runs([stats.training_loss for stats in training_stats_list])
    socs_model = None
    socs_no_model = None
    dist_table_differences = None

    if all(stats.used_best_mode for stats in training_stats_list):
        socs_model = _align_runs([stats.socs_model for stats in training_stats_list])
        socs_no_model = _align_runs(
            [stats.socs_no_model for stats in training_stats_list]
        )

    if all(stats.has_dist_table_differences for stats in training_stats_list):
        dist_table_differences = _align_runs(
            [stats.dist_table_differences for stats in training_stats_list]
        )

    num_epochs = min(stats.epochs for stats in training_stats_list)
    _plot_multiple_training_stats(
        num_epochs=num_epochs,
        socs=aligned_socs,
        losses=aligned_losses,
        socs_model=socs_model,
        socs_no_model=socs_no_model,
        dist_table_differences=dist_table_differences,
    )


def _plot_training_stats(
    num_epochs: int,
    socs: List[int],
    losses: List[float],
    socs_model: List[int] | None = None,
    socs_no_model: List[int] | None = None,
    dist_table_differences: List[float] | None = None,
    show: bool = True,
) -> None:
    """
    Plot training statistics of a single training run.

    Args:
        num_epochs (int): Number of training epochs.
        socs (List[int]): List of SOCs per epoch.
        losses (List[float]): List of losses per epoch.
        stats_paths (List[str]): List of paths to the stats files.
    """
    plt.figure(figsize=(12, 5))
    epochs = list(range(1, num_epochs + 1))

    num_columns: int = 2
    used_dist_table_differences: bool = dist_table_differences is not None
    if used_dist_table_differences:
        num_columns = 3

    plt.subplot(1, num_columns, 1)
    plt.plot(
        epochs,
        socs,
        color=SOC_PLOTTING_PARAMS["color"],
        marker=SOC_PLOTTING_PARAMS["marker"],
        label=SOC_PLOTTING_PARAMS["label"],
    )
    if socs_model is not None and socs_no_model is not None:
        plt.plot(
            epochs,
            socs_model,
            color=SOC_MODEL_PLOTTING_PARAMS["color"],
            marker=SOC_MODEL_PLOTTING_PARAMS["marker"],
            label=SOC_MODEL_PLOTTING_PARAMS["label"],
        )
        plt.plot(
            epochs,
            socs_no_model,
            color=SOC_NO_MODEL_PLOTTING_PARAMS["color"],
            marker=SOC_NO_MODEL_PLOTTING_PARAMS["marker"],
            label=SOC_NO_MODEL_PLOTTING_PARAMS["label"],
        )

    plt.title("Sum of Costs (SOC) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SOC")
    plt.legend()

    plt.subplot(1, num_columns, 2)
    plt.plot(
        epochs,
        losses,
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
    socs_model: List[List[int]] | None = None,
    socs_no_model: List[List[int]] | None = None,
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

    mean_socs, min_socs, max_socs = _aggregate(socs)
    plt.plot(
        epochs,
        mean_socs,
        color=SOC_PLOTTING_PARAMS["color"],
        linewidth=2,
        label=SOC_PLOTTING_PARAMS["label_multiple"],
    )
    if socs_model is not None and socs_no_model is not None:
        soc_model = _align_runs(socs_model)
        soc_no_model = _align_runs(socs_no_model)
        mean_soc_model, min_soc_model, max_soc_model = _aggregate(soc_model)
        mean_soc_no_model, min_soc_no_model, max_soc_no_model = _aggregate(soc_no_model)
        plt.fill_between(
            epochs,
            min_soc_model,
            max_soc_model,
            alpha=0.2,
            color="lightgreen",
            label="SOC with Model range",
        )
        plt.fill_between(
            epochs,
            min_soc_no_model,
            max_soc_no_model,
            alpha=0.2,
            color="lightcoral",
            label="SOC without Model range",
        )
        plt.plot(
            epochs,
            mean_soc_model,
            color=SOC_MODEL_PLOTTING_PARAMS["color"],
            linewidth=2,
            label=SOC_MODEL_PLOTTING_PARAMS["label_multiple"],
        )
        plt.plot(
            epochs,
            mean_soc_no_model,
            color=SOC_NO_MODEL_PLOTTING_PARAMS["color"],
            linewidth=2,
            label=SOC_NO_MODEL_PLOTTING_PARAMS["label_multiple"],
        )
    else:
        plt.fill_between(
            epochs, min_socs, max_socs, alpha=0.2, color="skyblue", label="SOC range"
        )

    plt.title("Sum of Costs (SOC) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SOC")
    plt.legend()

    plt.subplot(1, num_columns, 2)

    mean_losses, _, _ = _aggregate(losses)
    for idx, loss in enumerate(losses):
        plt.plot(
            epochs,
            loss,
            color="orange",
            alpha=0.3,
            linewidth=1,
            label="Run losses" if idx == 0 else None,
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
        ) = _aggregate(dist_table_differences)
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
            label="Distance Table Differences range",
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


if __name__ == "__main__":
    main()
