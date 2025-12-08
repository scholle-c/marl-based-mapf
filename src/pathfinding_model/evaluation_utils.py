"""
    Some helpful methods for visualizing the result of the reinforcement-learning training
"""
import json
import matplotlib.pyplot as plt
import argparse


def plot_training_stats(stats_filepath: str) -> None:
    with open(stats_filepath, "r") as f:
        stats = json.load(f)

    socs = stats["socs"]
    losses = stats["losses"]

    epochs = list(range(1, len(socs) + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, socs, marker='o')
    plt.title("Sum of Costs (SOC) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SOC")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, losses, marker='o', color='orange')
    plt.title("Mean Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training statistics for distance table CNN model.")
    parser.add_argument(
        "--stats-file",
        type=str,
        required=True,
        help="Path to the JSON file containing training statistics.",
    )
    args = parser.parse_args()
    plot_training_stats(args.stats_file)