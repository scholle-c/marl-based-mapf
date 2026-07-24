"""
Is used when you want to use a trained model for inference. Contains functions
to load a model and run a forward pass.
"""

from __future__ import annotations

from .definition import DefaultModel, DistanceTableCNN, PatchTransformer, FovPatchTransformer
from .feature_extraction import (
    FeatureExtractor,
    BasicExtractor,
    BinaryAgentsChannelExtractor,
    AggregatedAgentsChannelExtractor,
    RichAgentsChannelExtractor,
    CollisionAwareAgentsChannelExtractor,
    PathMembershipAgentsChannelExtractor,
    FovPathExtractor,
)
from marl_path.shared.mapf_utils import BfsCache
import torch
import numpy as np
from typing import Any


def save_checkpoint(
    model: DefaultModel, extractor: FeatureExtractor, path: str
) -> None:
    """Save model weights, extractor config, and model architecture together."""
    if isinstance(model, FovPatchTransformer):
        model_config = {
            "arch": "fov_transformer",
            "in_channels": extractor.n_channels,
            "vit_embed_dim": model._embed_dim,
            "vit_layers": model._num_layers,
            "vit_heads": model._num_heads,
        }
    elif isinstance(model, PatchTransformer):
        model_config = {
            "arch": "vit",
            "in_channels": extractor.n_channels,
            "grid_height": model._grid_height,
            "grid_width": model._grid_width,
            "vit_patch_size": model._patch_size,
            "vit_embed_dim": model._embed_dim,
            "vit_layers": model._num_layers,
            "vit_heads": model._num_heads,
        }
    else:
        model_config = {
            "arch": "cnn",
            "in_channels": extractor.n_channels,
            "hidden_channels": getattr(model, "_hidden_channels", None),
            "depth": getattr(model, "_depth", None),
        }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "extractor": {
                "class": type(extractor).__name__,
                "use_coord_channels": getattr(extractor, "_use_coord_channels", True),
                "agents_filter": getattr(extractor, "_agents_filter", None),
                "include_intersection": getattr(
                    extractor, "_include_intersection", None
                ),
                "encode_time": getattr(extractor, "_encode_time", None),
                "fov_radius": getattr(extractor, "_fov_radius", None),
            },
            "model_config": model_config,
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
        model_config = checkpoint.get("model_config", {})
        in_channels = model_config.get("in_channels", extractor.n_channels)
        arch = model_config.get("arch", "cnn")
    else:
        # Legacy format: bare state_dict saved with torch.save(model.state_dict(), path)
        state_dict = checkpoint
        extractor = BasicExtractor()
        in_channels = 5
        model_config = {}
        arch = "cnn"

    if arch == "fov_transformer":
        model = FovPatchTransformer(
            in_channels=in_channels,
            embed_dim=model_config.get("vit_embed_dim") or 64,
            num_layers=model_config.get("vit_layers") or 4,
            num_heads=model_config.get("vit_heads") or 4,
        ).to(device)
    elif arch == "vit":
        model = PatchTransformer(
            in_channels=in_channels,
            grid_height=model_config.get("grid_height", 32),
            grid_width=model_config.get("grid_width", 32),
            patch_size=model_config.get("vit_patch_size") or 1,
            embed_dim=model_config.get("vit_embed_dim") or 64,
            num_layers=model_config.get("vit_layers") or 4,
            num_heads=model_config.get("vit_heads") or 4,
        ).to(device)
    else:
        model = DistanceTableCNN(
            in_channels=in_channels,
            hidden_channels=model_config.get("hidden_channels") or 32,
            depth=model_config.get("depth") or 4,
        ).to(device)
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
    other_starts: list[tuple[int, int]] | None = None,
    bfs_cache: BfsCache | None = None,
) -> Any:
    """
    Predict a distance/value table for the provided map, start, and goal.

    Parameters
    ----------
    start:
        This agent's start coordinate as (y, x) in grid coordinates.
    goal:
        This agent's goal coordinate as (y, x) in grid coordinates.
    grid:
        2D map array where non-zero entries denote traversable cells.
    extractor:
        Feature extractor to use. Defaults to BasicExtractor.
    other_agents:
        Other agents' GOAL positions. Defaults to empty list.
    other_starts:
        Other agents' START positions, index-aligned with `other_agents`.
        Needed (together with `bfs_cache`) by extractors that reconstruct
        individual agent paths, e.g. PathMembershipAgentsChannelExtractor —
        without it those extractors silently fall back to all-zero path
        channels, which is out-of-distribution for a model trained via
        CbsDataset (which always supplies these). Defaults to empty list.
    bfs_cache:
        Reuse an existing BfsCache for this grid instead of building a fresh
        one (BFS itself is cheap, but this avoids redundant recomputation
        across repeated calls on the same map).
    """
    if extractor is None:
        extractor = BasicExtractor()
    if other_agents is None:
        other_agents = []
    if other_starts is None:
        other_starts = []

    grid_np = np.asarray(grid)
    if grid_np.ndim != 2:
        raise ValueError("Grid must be a 2D array.")

    if bfs_cache is None:
        bfs_cache = BfsCache(grid_np)
    bfs_tables = {g: bfs_cache[g] for g in other_agents}
    bfs_tables.update({s: bfs_cache[s] for s in other_starts})
    bfs_tables[goal] = bfs_cache[goal]

    device = next(model.parameters()).device

    if isinstance(extractor, FovPathExtractor):
        tokens = extractor.extract_tokens(
            grid_np,
            goal,
            other_agents,
            other_starts=other_starts,
            own_start=start,
            bfs_tables=bfs_tables,
            device=device,
        )
        prediction = np.zeros(grid_np.shape, dtype=np.float32)
        if tokens.coords.shape[0] > 0:
            with torch.no_grad():
                probs = (
                    model(tokens.features.unsqueeze(0), tokens.coords.unsqueeze(0))
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
            ys = tokens.coords[:, 0].cpu().numpy()
            xs = tokens.coords[:, 1].cpu().numpy()
            prediction[ys, xs] = probs
        return prediction

    input_tensor = extractor.extract(
        grid_np,
        goal,
        start,
        other_agents,
        device=device,
        bfs_tables=bfs_tables,
        other_starts=other_starts,
        own_start=start,
    )

    with torch.no_grad():
        prediction = model(input_tensor).squeeze().cpu().numpy()
    return prediction


def _extractor_from_config(config: dict | None) -> FeatureExtractor:
    if config is None:
        return BasicExtractor()
    use_coord = config.get("use_coord_channels", True)
    cls_name = config.get("class", "BasicExtractor")
    if cls_name == "BinaryAgentsChannelExtractor":
        return BinaryAgentsChannelExtractor(use_coord_channels=use_coord)
    if cls_name == "AggregatedAgentsChannelExtractor":
        return AggregatedAgentsChannelExtractor(use_coord_channels=use_coord)
    if cls_name == "RichAgentsChannelExtractor":
        return RichAgentsChannelExtractor(use_coord_channels=use_coord)
    if cls_name == "CollisionAwareAgentsChannelExtractor":
        return CollisionAwareAgentsChannelExtractor(use_coord_channels=use_coord)
    if cls_name == "PathMembershipAgentsChannelExtractor":
        agents_filter = config.get("agents_filter") or "all"
        return PathMembershipAgentsChannelExtractor(
            agents_filter=agents_filter,
            include_intersection=bool(config.get("include_intersection")),
            encode_time=bool(config.get("encode_time")),
            use_coord_channels=use_coord,
        )
    if cls_name == "FovPathExtractor":
        return FovPathExtractor(
            fov_radius=config.get("fov_radius") or 2,
            include_intersection=bool(config.get("include_intersection")),
        )
    return BasicExtractor(use_coord_channels=use_coord)
