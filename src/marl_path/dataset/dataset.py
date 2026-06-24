"""PyTorch Dataset over cached CBS instances."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from marl_path.shared.mapf_utils import BfsCache, get_grid, get_scenario
from marl_path.model.feature_extraction import BasicExtractor, FeatureExtractor
from marl_path.model.training import (
    DelayBatchItem,
    _get_path_target_first_visit,
    _get_via_coordinates,
)
from .instance import CachedInstance


class CbsDataset(Dataset):
    """Dataset of CBS-optimal MAPF solutions for supervised delay training.

    Each item is a DelayBatchItem for one instance (all agents), directly
    compatible with update_delay_from_batch.

    Input tensors are built at load time via the extractor so swapping
    extractors does not require regenerating the cache.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        extractor: Optional[FeatureExtractor] = None,
        device: Optional[torch.device] = None,
    ):
        self._cache_dir = Path(cache_dir)
        self._extractor = extractor or BasicExtractor()
        self._device = device or torch.device("cpu")
        self._files = sorted(self._cache_dir.glob("*.npz"))
        if not self._files:
            raise ValueError(f"No .npz files found in {cache_dir}")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> DelayBatchItem:
        instance = CachedInstance.load(self._files[idx])
        grid = get_grid(instance.map_file)
        # agent_indices may be a non-contiguous subset, so load all and select.
        _all_starts, all_scen_goals = get_scenario(instance.scen_file)
        bfs_cache = BfsCache(grid)

        # Mirrors DistTable.compute_delay_model: other_agents = other agents' goals,
        # and the "start" slot in the extractor receives this agent's own goal.
        all_goals = [all_scen_goals[i] for i in instance.agent_indices]

        input_tensors: list[torch.Tensor] = []
        bfs_distances: list[np.ndarray] = []
        all_targets: list[float] = []

        for i, path in enumerate(instance.paths):
            goal = all_goals[i]
            other_goals = [all_goals[j] for j in range(len(all_goals)) if j != i]
            bfs_table = bfs_cache[goal]
            other_bfs = {g: bfs_cache[g] for g in other_goals}

            tensor = self._extractor.extract(
                grid,
                goal=goal,
                start=goal,
                other_agents=other_goals,
                device=self._device,
                bfs_tables=other_bfs,
            )
            input_tensors.append(tensor)

            bfs_distances.append(_get_via_coordinates(bfs_table, path))
            all_targets.extend(_get_path_target_first_visit(path))

        target_tensor = torch.tensor(all_targets, dtype=torch.float32, device=self._device)
        return DelayBatchItem(input_tensors, instance.paths, bfs_distances, target_tensor)
