from pathlib import Path
import json
from typing import Iterable, Sequence
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from torchviz import make_dot

from value_map_learner import DistanceTableCNN
from value_map_learner.mapf_utils import get_grid


def plot_loss_curve(
    history_path: str | Path = "output/loss_history.json",
    output_path: str | Path = "output/loss_curve.png",
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
    plt.close()
    print(f"Saved loss curve to {output_path.resolve()}")


def _prepare_model(model_path: str | Path, device: torch.device) -> DistanceTableCNN:
    model = DistanceTableCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_input_tensor(grid: np.ndarray, goal: tuple[int, int]) -> torch.Tensor:
    if grid.ndim != 2:
        raise ValueError("Grid must be a 2D array.")
    if not grid[goal]:
        raise ValueError(f"Goal {goal} is not accessible in the provided map.")

    map_channel = grid.astype(np.float32, copy=False)
    goal_channel = np.zeros_like(map_channel, dtype=np.float32)
    goal_channel[goal] = 1.0
    start_channel = np.zeros_like(map_channel, dtype=np.float32)
    free_positions = np.argwhere(grid)
    if free_positions.size == 0:
        raise ValueError("No free cells available in the map to sample a start position.")
    start_idx = np.random.randint(len(free_positions))
    start_y, start_x = free_positions[start_idx]
    start_channel[start_y, start_x] = 1.0
    stacked = np.stack((map_channel, goal_channel, start_channel), axis=0)
    return torch.from_numpy(stacked).unsqueeze(0)  # (1, 3, H, W)


def plot_predicted_value_map(
    map_path: str | Path,
    goal: Sequence[int] | tuple[int, int],
    model_path: str | Path = "output/model.pt",
    device: str | torch.device | None = None,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """
    Load a saved model, predict a distance/value map for a given goal, and plot it.

    Parameters
    ----------
    map_path:
        Path to the map file readable by `get_grid`.
    goal:
        Goal coordinate as (y, x) in grid coordinates.
    model_path:
        Path to the saved model checkpoint (state_dict).
    device:
        Torch device; defaults to CUDA if available else CPU.
    output_path:
        If provided, save the plot to this path; otherwise shows the plot interactively.

    Returns
    -------
    np.ndarray
        Predicted value map (H, W) as a numpy array.
    """
    goal_tuple = (int(goal[0]), int(goal[1]))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    grid = get_grid(map_path)
    model = _prepare_model(model_path, device)
    input_tensor = _build_input_tensor(grid, goal_tuple).to(device)

    with torch.no_grad():
        prediction = model(input_tensor).cpu().squeeze().numpy()

    plt.figure(figsize=(6, 5))
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad(color="lightgray")

    obstacle_mask = ~grid.astype(bool)
    display_data = np.ma.array(prediction, mask=obstacle_mask)

    im = plt.imshow(display_data, cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Predicted value map\n{Path(map_path).name}, goal={goal_tuple}")

    # overlay numeric values on free cells
    for (y, x), value in np.ndenumerate(prediction):
        if obstacle_mask[y, x]:
            continue
        text_color = "white" if value > np.nanmedian(prediction) else "black"
        plt.text(x, y, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=7)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        print(f"Saved predicted value map to {output_path.resolve()}")
    else:
        plt.tight_layout()
        plt.show()
    plt.close()

    return prediction


def visualize_model(
    model_path: str | Path = "output/model.pt",
    output_path: str | Path = "output/model_graph",
    map_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> None:
    """
    Visualize the DistanceTableCNN computation graph using torchviz.

    Parameters
    ----------
    model_path:
        Path to the saved model checkpoint (state_dict).
    output_path:
        Output path (without extension) for the graph; torchviz appends format.
    map_path:
        Map file used to infer the input height/width for the dummy tensor.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DistanceTableCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if map_path is None:
        raise ValueError("map_path is required to match the dummy input size to the map.")

    height, width = get_grid(map_path).shape
    dummy = torch.randn(1, 3, height, width, device=device)
    output = model(dummy)
    graph = make_dot(output, params=dict(model.named_parameters()))
    graph.render(Path(output_path), format="png", cleanup=True)
    print(f"Saved model graph to {Path(output_path).with_suffix('.png').resolve()}")


def main() -> None:
    """CLI helper to plot loss curve, predict a value map, or visualize the model graph based on arguments."""
    parser = argparse.ArgumentParser(description="Evaluation utilities for Value Map Learner.")
    parser.add_argument(
        "--mode",
        choices=["loss", "predict", "visualize"],
        default="loss",
        help="Choose 'loss' to plot training curves, 'predict' to plot a value map, or 'visualize' to export the model graph.",
    )
    parser.add_argument("--history-path", default="output/loss_history.json", help="Path to loss history JSON.")
    parser.add_argument("--loss-output", default="output/loss_curve.png", help="Output path for loss plot.")
    parser.add_argument("--map-path", help="Map file path for prediction/visualization.")
    parser.add_argument("--goal", nargs=2, type=int, metavar=("Y", "X"), help="Goal coordinate (y x).")
    parser.add_argument("--model-path", default="output/model.pt", help="Checkpoint path for prediction.")
    parser.add_argument("--pred-output", help="Output path for predicted value map image.")
    parser.add_argument("--graph-output", default="output/model_graph", help="Output path for model graph image.")
    parser.add_argument("--device", help="Device for model inference (e.g., cuda or cpu).")

    args = parser.parse_args()

    if args.mode == "loss":
        plot_loss_curve(history_path=args.history_path, output_path=args.loss_output)
    elif args.mode == "predict":
        if not args.map_path or not args.goal:
            parser.error("--map-path and --goal are required for mode 'predict'.")
        plot_predicted_value_map(
            map_path=args.map_path,
            goal=tuple(args.goal),
            model_path=args.model_path,
            device=args.device,
            output_path=args.pred_output,
        )
    else:
        if not args.map_path:
            parser.error("--map-path is required for mode 'visualize'.")
        visualize_model(
            model_path=args.model_path,
            output_path=args.graph_output,
            map_path=args.map_path,
            device=args.device,
        )


if __name__ == "__main__":
    main()
