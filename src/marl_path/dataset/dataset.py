"""PyTorch Dataset over cached CBS instances."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from marl_path.shared.mapf_utils import BfsCache, get_grid, get_scenario
from marl_path.delay_methods import DelayMethod
from marl_path.model.feature_extraction import BasicExtractor, FeatureExtractor, FovPathExtractor
from marl_path.model.training import (
    DenseDelayBatchItem,
    prepare_dense_delay_batch_item,
    FovDelayBatchItem,
    prepare_fov_delay_batch_item,
)
from .instance import CachedInstance


class CbsDataset(Dataset):
    """Dataset of CBS-optimal MAPF solutions for supervised delay training.

    Each item is a DenseDelayBatchItem: a full-grid, per-agent delay target
    built by `delay_method` (default NonOptimalPenaltyDelay).

    Input tensors are built at load time via the extractor so swapping
    extractors does not require regenerating the cache.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        extractor: Optional[FeatureExtractor] = None,
        device: Optional[torch.device] = None,
        delay_method: Optional[DelayMethod] = None,
    ):
        self._cache_dir = Path(cache_dir)
        self._extractor = extractor or BasicExtractor()
        self._device = device or torch.device("cpu")
        self._delay_method = delay_method
        self._files = sorted(self._cache_dir.glob("*.npz"))
        if not self._files:
            raise ValueError(f"No .npz files found in {cache_dir}")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> DenseDelayBatchItem | FovDelayBatchItem:
        instance = CachedInstance.load(self._files[idx])
        grid = get_grid(instance.map_file)
        # agent_indices may be a non-contiguous subset, so load all and select.
        all_scen_starts, all_scen_goals = get_scenario(instance.scen_file)
        bfs_cache = BfsCache(grid)

        # Mirrors DistTable.compute_delay_model: other_agents = other agents' goals,
        # and the "start"/goal slot in the extractor receives this agent's own goal
        # (the model predicts a delay field anchored at the goal, independent of the
        # agent's current position).
        all_goals = [all_scen_goals[i] for i in instance.agent_indices]
        all_starts = [all_scen_starts[i] for i in instance.agent_indices]

        if isinstance(self._extractor, FovPathExtractor):
            return self._getitem_fov(instance, grid, bfs_cache, all_goals, all_starts)
        return self._getitem_dense(instance, grid, bfs_cache, all_goals, all_starts)

    def _getitem_dense(
        self, instance, grid, bfs_cache, all_goals, all_starts
    ) -> DenseDelayBatchItem:
        input_tensors: list[torch.Tensor] = []
        for i in range(len(instance.paths)):
            goal = all_goals[i]
            other_goals = [all_goals[j] for j in range(len(all_goals)) if j != i]
            other_starts = [all_starts[j] for j in range(len(all_starts)) if j != i]
            other_bfs = {g: bfs_cache[g] for g in other_goals}
            other_bfs.update({s: bfs_cache[s] for s in other_starts})
            # Own goal's BFS table, needed by extractors that reconstruct this
            # agent's individual shortest path (e.g. CollisionAwareAgentsChannelExtractor).
            other_bfs[goal] = bfs_cache[goal]

            tensor = self._extractor.extract(
                grid,
                goal=goal,
                start=goal,
                other_agents=other_goals,
                device=self._device,
                bfs_tables=other_bfs,
                other_starts=other_starts,
                own_start=all_starts[i],
            )
            input_tensors.append(tensor)

        return prepare_dense_delay_batch_item(
            grid,
            bfs_cache,
            instance.paths,
            all_goals,
            self._device,
            input_tensors,
            self._delay_method,
        )

    def _getitem_fov(
        self, instance, grid, bfs_cache, all_goals, all_starts
    ) -> FovDelayBatchItem:
        tokens_per_agent = []
        for i in range(len(instance.paths)):
            goal = all_goals[i]
            other_goals = [all_goals[j] for j in range(len(all_goals)) if j != i]
            other_starts = [all_starts[j] for j in range(len(all_starts)) if j != i]
            other_bfs = {g: bfs_cache[g] for g in other_goals}
            other_bfs.update({s: bfs_cache[s] for s in other_starts})
            other_bfs[goal] = bfs_cache[goal]

            tokens = self._extractor.extract_tokens(
                grid,
                goal=goal,
                other_agents=other_goals,
                other_starts=other_starts,
                own_start=all_starts[i],
                bfs_tables=other_bfs,
                device=self._device,
            )
            tokens_per_agent.append(tokens)

        return prepare_fov_delay_batch_item(
            grid,
            bfs_cache,
            instance.paths,
            all_goals,
            self._device,
            tokens_per_agent,
            self._delay_method,
        )
