from pathlib import Path
import json

import matplotlib.pyplot as plt


def plot_loss_curve(
    history_path: str | Path = "artifacts/loss_history.json",
    output_path: str | Path = "artifacts/loss_curve.png",
) -> None:
    """Load loss history JSON and save a loss curve image."""
    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    train_losses = history.get("train", [])
    val_losses = history.get("val", [])
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved loss curve to {output_path.resolve()}")


def main() -> None:
    """CLI entrypoint to plot loss curves using default paths."""
    plot_loss_curve()


if __name__ == "__main__":
    main()
