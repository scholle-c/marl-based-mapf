"""
    Some helpful methods for visualizing the result of the reinforcement-learning training
"""
import argparse
import json
import os
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt


def _load_stats(path: str) -> Tuple[List[float], List[float]]:
    """
    Returns (socs, losses) from a stats JSON file. If a directory is passed,
    it looks for a training_stats.json inside it.
    """
    target_path = path
    if os.path.isdir(path):
        target_path = os.path.join(path, "training_stats.json")
    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"Could not find stats file at {target_path}")

    with open(target_path, "r") as f:
        stats = json.load(f)
    return stats["socs"], stats["losses"]


def _align_runs(runs: Iterable[List[float]]) -> List[List[float]]:
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


def _aggregate(values: List[List[float]]) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate per-epoch mean, min and max across runs.
    """
    transposed = list(zip(*values))
    mean_values = [sum(v) / len(v) for v in transposed]
    min_values = [min(v) for v in transposed]
    max_values = [max(v) for v in transposed]
    return mean_values, min_values, max_values


def plot_training_stats(stats_paths: List[str]) -> None:
    soc_runs = []
    loss_runs = []
    for path in stats_paths:
        socs, losses = _load_stats(path)
        soc_runs.append(socs)
        loss_runs.append(losses)

    single_run = len(soc_runs) == 1
    soc_runs = _align_runs(soc_runs)
    loss_runs = _align_runs(loss_runs)
    epochs = list(range(1, len(soc_runs[0]) + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if single_run:
        plt.plot(epochs, soc_runs[0], marker="o", label="SOC")
    else:
        mean_socs, min_socs, max_socs = _aggregate(soc_runs)
        plt.fill_between(epochs, min_socs, max_socs, alpha=0.2, color="skyblue", label="SOC range")
        plt.plot(epochs, mean_socs, color="blue", linewidth=2, label="Mean SOC")
    plt.title("Sum of Costs (SOC) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SOC")
    plt.legend()

    plt.subplot(1, 2, 2)
    if single_run:
        plt.plot(epochs, loss_runs[0], marker="o", color="orange", label="Loss")
    else:
        mean_losses, _, _ = _aggregate(loss_runs)
        for idx, losses in enumerate(loss_runs):
            plt.plot(epochs, losses, color="orange", alpha=0.3, linewidth=1, label="Run losses" if idx == 0 else None)
        plt.plot(epochs, mean_losses, color="red", linewidth=2, label="Mean Loss")
    plt.title("Mean Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training statistics for distance table CNN model.")
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
    plot_training_stats(args.stats_file)
