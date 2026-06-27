"""One-file-per-instance cache of a CBS-optimal MAPF solution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from marl_path.shared import Coord


@dataclass
class CachedInstance:
    """Minimal cache for one MAPF instance solved by CBS.

    Stores raw paths + metadata only. Input tensors are reconstructed at
    load time so that changing the feature extractor does not invalidate the cache.
    """

    map_file: str
    scen_file: str
    agent_indices: list[int]
    paths: list[list[Coord]]

    def save(self, path: Path) -> None:
        """Serialize to a .npz file."""
        lengths = np.array([len(p) for p in self.paths], dtype=np.int32)
        flat = (
            np.array([(y, x) for p in self.paths for y, x in p], dtype=np.int32)
            if any(self.paths)
            else np.empty((0, 2), dtype=np.int32)
        )
        np.savez(
            path,
            path_lengths=lengths,
            path_coords=flat,
            agent_indices=np.array(self.agent_indices, dtype=np.int32),
            map_file=np.array([self.map_file]),
            scen_file=np.array([self.scen_file]),
        )

    @classmethod
    def load(cls, path: Path) -> CachedInstance:
        """Deserialize from a .npz file."""
        data = np.load(path, allow_pickle=False)
        lengths = data["path_lengths"]
        coords = data["path_coords"]
        paths: list[list[Coord]] = []
        offset = 0
        for n in lengths:
            chunk = coords[offset : offset + int(n)]
            paths.append([(int(r), int(c)) for r, c in chunk])
            offset += int(n)
        return cls(
            map_file=str(data["map_file"][0]),
            scen_file=str(data["scen_file"][0]),
            agent_indices=data["agent_indices"].tolist(),
            paths=paths,
        )
