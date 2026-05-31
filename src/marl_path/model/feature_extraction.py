"""Feature extraction strategies for MAPF input tensor construction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from marl_path.shared import Grid


class FeatureExtractor(ABC):
    """Base class for MAPF input feature extraction strategies."""

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of channels produced by this extractor."""
        ...

    @abstractmethod
    def extract(
        self,
        grid: Grid,
        goal: tuple[int, int],
        start: tuple[int, int],
        other_agents: list[tuple[int, int]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return a tensor of shape (1, n_channels, H, W)."""
        ...


def _add_relative_coords(tensor: torch.Tensor, goal: tuple[int, int]) -> torch.Tensor:
    """Append two relative coordinate channels (y, x) w.r.t. goal to (B, C, H, W)."""
    batch_size, _, height, width = tensor.shape
    target_y, target_x = goal
    y_rel = (torch.arange(height, dtype=torch.float32) - target_y) / height
    x_rel = (torch.arange(width, dtype=torch.float32) - target_x) / width
    y_coords = y_rel.view(1, 1, height, 1).expand(batch_size, 1, height, width)
    x_coords = x_rel.view(1, 1, 1, width).expand(batch_size, 1, height, width)
    y_coords = y_coords.to(tensor.device)
    x_coords = x_coords.to(tensor.device)
    return torch.cat([tensor, y_coords, x_coords], dim=1)


class BasicExtractor(FeatureExtractor):
    """Map + goal + start channels with optional relative coordinate channels.

    Reproduces the behaviour of build_input_tensor. Other agents are ignored.
    Channels: map, goal, start [, y_rel, x_rel]
    """

    def __init__(self, use_coord_channels: bool = True):
        self._use_coord_channels = use_coord_channels

    @property
    def n_channels(self) -> int:
        return 5 if self._use_coord_channels else 3

    def extract(
        self,
        grid: Grid,
        goal: tuple[int, int],
        start: tuple[int, int],
        other_agents: list[tuple[int, int]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        map_ch = grid.astype(np.float32, copy=False)
        goal_ch = np.zeros_like(map_ch)
        goal_ch[goal] = 1.0
        start_ch = np.zeros_like(map_ch)
        start_ch[start] = 1.0

        tensor = torch.from_numpy(np.stack([map_ch, goal_ch, start_ch])).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        if self._use_coord_channels:
            tensor = _add_relative_coords(tensor, goal)
        return tensor


class OtherAgentsChannelExtractor(FeatureExtractor):
    """BasicExtractor plus one binary channel marking all other agent positions.

    Channels: map, goal, start, other_agents [, y_rel, x_rel]
    """

    def __init__(self, use_coord_channels: bool = True):
        self._use_coord_channels = use_coord_channels

    @property
    def n_channels(self) -> int:
        return 6 if self._use_coord_channels else 4

    def extract(
        self,
        grid: Grid,
        goal: tuple[int, int],
        start: tuple[int, int],
        other_agents: list[tuple[int, int]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        map_ch = grid.astype(np.float32, copy=False)
        goal_ch = np.zeros_like(map_ch)
        goal_ch[goal] = 1.0
        start_ch = np.zeros_like(map_ch)
        start_ch[start] = 1.0
        agents_ch = np.zeros_like(map_ch)
        for pos in other_agents:
            agents_ch[pos] = 1.0

        tensor = torch.from_numpy(
            np.stack([map_ch, goal_ch, start_ch, agents_ch])
        ).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        if self._use_coord_channels:
            tensor = _add_relative_coords(tensor, goal)
        return tensor
