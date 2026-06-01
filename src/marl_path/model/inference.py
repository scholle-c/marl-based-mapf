"""
Is used when you want to use a trained model for inference. Contains functions
to load a model and run a forward pass.
"""

from __future__ import annotations

from .definition import DefaultModel, DistanceTableCNN
from .feature_extraction import (
    FeatureExtractor,
    BasicExtractor,
    OtherAgentsChannelExtractor,
)
import torch
import numpy as np
from typing import Any


def save_checkpoint(
    model: DefaultModel, extractor: FeatureExtractor, path: str
) -> None:
    """Save model weights and extractor config together."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "extractor": {
                "class": type(extractor).__name__,
                "use_coord_channels": getattr(extractor, "_use_coord_channels", True),
            },
            "model_config": {"in_channels": extractor.n_channels},
        },
        path,
    )


def load_model(
    model_path: str, device: torch.device | None = None
) -> tuple[DefaultModel, FeatureExtractor]:
    """Load a trained model and its extractor from a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        extractor = _extractor_from_config(checkpoint.get("extractor"))
        in_channels = checkpoint.get("model_config", {}).get(
            "in_channels", extractor.n_channels
        )
    else:
        # Legacy format: bare state_dict saved with torch.save(model.state_dict(), path)
        state_dict = checkpoint
        extractor = BasicExtractor()
        in_channels = 5
    model = DistanceTableCNN(in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, extractor


def predict_distance_table(
    model: DistanceTableCNN,
    grid: Any,
    start: tuple[int, int],
    goal: tuple[int, int],
    extractor: FeatureExtractor | None = None,
    other_agents: list[tuple[int, int]] | None = None,
) -> Any:
    """
    Predict a distance/value table for the provided map, start, and goal.

    Parameters
    ----------
    start:
        Start coordinate as (y, x) in grid coordinates.
    goal:
        Goal coordinate as (y, x) in grid coordinates.
    grid:
        2D map array where non-zero entries denote traversable cells.
    extractor:
        Feature extractor to use. Defaults to BasicExtractor.
    other_agents:
        Positions of other agents to encode. Defaults to empty list.
    """
    if extractor is None:
        extractor = BasicExtractor()
    if other_agents is None:
        other_agents = []

    grid_np = np.asarray(grid)
    if grid_np.ndim != 2:
        raise ValueError("Grid must be a 2D array.")

    device = next(model.parameters()).device
    input_tensor = extractor.extract(grid_np, goal, start, other_agents, device=device)

    with torch.no_grad():
        prediction = model(input_tensor).squeeze().cpu().numpy()
    return prediction


def _extractor_from_config(config: dict | None) -> FeatureExtractor:
    if config is None:
        return BasicExtractor()
    use_coord = config.get("use_coord_channels", True)
    cls_name = config.get("class", "BasicExtractor")
    if cls_name == "OtherAgentsChannelExtractor":
        return OtherAgentsChannelExtractor(use_coord_channels=use_coord)
    return BasicExtractor(use_coord_channels=use_coord)
