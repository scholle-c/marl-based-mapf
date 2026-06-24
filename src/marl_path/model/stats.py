"""Module for storing and handling training statistics."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np
from importlib.metadata import version

import marl_path.constants as consts

__version__ = version("marl-path")

STORE_LACAM_TABLE_ONLY_ONCE: bool = True

_FILENAME_METRICS = "metrics.csv"
_FILENAME_CONFIG = "config.json"


@dataclass
class MAPFStats:
    """MAPF instance metadata and per-epoch solution quality."""

    socs: List[int | float | None] = field(default_factory=list)
    elapsed_times: List[float | None] = field(default_factory=list)
    num_agents: int | None = None
    map_size: Tuple | None = None


@dataclass
class DistTableStats:
    """Optional heavy-data tracking for distance table comparison.

    record_mode: 0 = off, 1 = all agents, 2 = one agent only
    record_granularity: save full tables every N epochs
    """

    record_mode: int = 0
    record_granularity: int = 10
    _diffs: List[float] = field(default_factory=list, repr=False)
    _recorded_epochs: List[int] = field(default_factory=list, repr=False)
    _model_tables: List = field(default_factory=list, repr=False)
    _lacam_tables: List = field(default_factory=list, repr=False)

    def record(
        self,
        epoch: int,
        model_tables: List,
        lacam_tables: List | None,
    ) -> None:
        """Record dist table data for one epoch."""
        if lacam_tables is not None:
            diff = float(
                np.mean(
                    [
                        np.abs(m - lac).mean()
                        for m, lac in zip(model_tables, lacam_tables)
                    ]
                )
            )
            self._diffs.append(diff)
        else:
            self._diffs.append(float("nan"))

        if self.record_mode == 0 or (epoch - 1) % self.record_granularity != 0:
            return

        self._recorded_epochs.append(epoch)
        if self.record_mode == 1:
            self._model_tables.append(model_tables)
            self._maybe_add_lacam(epoch, lacam_tables)
        elif self.record_mode == 2:
            self._model_tables.append([model_tables[0]])
            self._maybe_add_lacam(epoch, lacam_tables, one_agent=True)

    def _maybe_add_lacam(
        self, epoch: int, lacam_tables: List | None, one_agent: bool = False
    ) -> None:
        if STORE_LACAM_TABLE_ONLY_ONCE and epoch > 1:
            return
        if lacam_tables is None:
            return
        self._lacam_tables.append([lacam_tables[0]] if one_agent else lacam_tables)

    def latest_diff(self) -> float | None:
        return self._diffs[-1] if self._diffs else None

    def save(self, output_dir: str) -> None:
        """Write full dist tables to CSV (rows = recorded epochs, cols = flattened H×W × agents)."""

        def _stack(tables_by_epoch: List) -> np.ndarray:
            return np.vstack([np.hstack(t) for t in tables_by_epoch])

        if self._lacam_tables:
            np.savetxt(
                os.path.join(output_dir, consts.DEFAULT_FILENAME_DIST_TABLE_LACAM),
                _stack(self._lacam_tables),
                delimiter=",",
                fmt="%.2f",
            )
        if self._model_tables:
            np.savetxt(
                os.path.join(output_dir, consts.DEFAULT_FILENAME_DIST_TABLE_MODEL),
                _stack(self._model_tables),
                delimiter=",",
                fmt="%.2f",
            )


@dataclass
class TrainingStats:
    """Per-epoch training statistics with structured CSV + JSON storage.

    Storage layout (written by save()):
        metrics.csv  — epoch, loss, soc[, dist_table_diff]  one row per epoch
        config.json  — metadata: mode, device, seed, map_size, num_agents
        dist_tables_model.csv / dist_tables_lacam.csv  — optional, sparse
        map_mask.csv — optional static grid
    """

    training_mode: str = ""
    used_device: str | None = None
    used_seed: int | None = None
    mapf: MAPFStats | None = None
    dist_tables: DistTableStats | None = None

    _epoch_count: int = field(default=0, repr=False)
    _losses: List[float | None] = field(default_factory=list, repr=False)
    _val_losses: List[float | None] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record_epoch(
        self,
        loss: float | None,
        soc: int | float | None = None,
        elapsed_time: float | None = None,
        val_loss: float | None = None,
    ) -> None:
        """Record scalar metrics for one training epoch."""
        self._epoch_count += 1
        self._losses.append(loss)
        self._val_losses.append(val_loss)
        if self.mapf is not None:
            self.mapf.socs.append(soc)
            self.mapf.elapsed_times.append(elapsed_time)

    def record_dist_tables(
        self,
        model_tables: List,
        lacam_tables: List | None = None,
    ) -> None:
        """Record dist table data for the current epoch. Call after record_epoch."""
        if self.dist_tables is not None:
            self.dist_tables.record(self._epoch_count, model_tables, lacam_tables)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def has_dist_table_differences(self) -> bool:
        return self.dist_tables is not None and len(self.dist_tables._diffs) > 0

    # ------------------------------------------------------------------ #
    # Saving                                                               #
    # ------------------------------------------------------------------ #

    def save(self, output_dir: str, map_mask: np.ndarray | None = None) -> None:
        """Write all stats to output_dir."""
        self._save_metrics_csv(output_dir)
        if self.dist_tables is not None and self.dist_tables.record_mode != 0:
            self.dist_tables.save(output_dir)
        if map_mask is not None:
            np.savetxt(
                os.path.join(output_dir, consts.DEFAULT_FILENAME_MAP_MASK),
                map_mask.astype(int),
                delimiter=",",
                fmt="%d",
            )

    def _save_metrics_csv(self, output_dir: str) -> None:
        socs: List[int | float | None] = (
            self.mapf.socs if self.mapf else [None] * self._epoch_count
        )
        elapsed_times: List[float | None] = (
            self.mapf.elapsed_times if self.mapf else [None] * self._epoch_count
        )
        has_diffs = (
            self.dist_tables is not None
            and len(self.dist_tables._diffs) == self._epoch_count
        )
        has_elapsed = len(elapsed_times) == self._epoch_count
        has_val_loss = any(v is not None for v in self._val_losses)
        fieldnames = ["epoch", "loss"]
        if has_val_loss:
            fieldnames.append("val_loss")
        fieldnames.append("soc")
        if has_elapsed:
            fieldnames.append("elapsed_time_ms")
        if has_diffs:
            fieldnames.append("dist_table_diff")

        with open(os.path.join(output_dir, _FILENAME_METRICS), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(self._epoch_count):
                row: dict[str, Any] = {
                    "epoch": i + 1,
                    "loss": self._losses[i],
                    "soc": socs[i] if i < len(socs) else None,
                }
                if has_val_loss:
                    row["val_loss"] = (
                        self._val_losses[i] if i < len(self._val_losses) else None
                    )
                if has_elapsed:
                    row["elapsed_time_ms"] = elapsed_times[i]
                if has_diffs:
                    row["dist_table_diff"] = self.dist_tables._diffs[i]  # type: ignore[union-attr]
                writer.writerow(row)

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, output_dir: str) -> TrainingStats:
        """Load stats from a directory written by save()."""
        config_path = os.path.join(output_dir, _FILENAME_CONFIG)
        config: dict = {}
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)
        else:
            used_config_path = os.path.join(output_dir, "used_config.json")
            if os.path.isfile(used_config_path):
                with open(used_config_path) as f:
                    config = json.load(f)

        mapf = None
        if "num_agents" in config:
            mapf = MAPFStats(
                num_agents=config.get("num_agents"),
                map_size=tuple(config["map_size"]) if config.get("map_size") else None,
            )

        stats = cls(
            training_mode=config.get("training_mode", ""),
            used_device=config.get("used_device"),
            used_seed=config.get("used_seed"),
            mapf=mapf,
        )

        with open(os.path.join(output_dir, _FILENAME_METRICS), newline="") as f:
            for row in csv.DictReader(f):
                loss_str = row["loss"]
                loss = float(loss_str) if loss_str not in ("", "None") else None
                soc_str = row.get("soc", "")
                soc = int(float(soc_str)) if soc_str not in ("", "None") else None
                elapsed_str = row.get("elapsed_time_ms", "")
                elapsed = (
                    float(elapsed_str) if elapsed_str not in ("", "None") else None
                )
                stats._epoch_count += 1
                stats._losses.append(loss)
                if stats.mapf is not None:
                    stats.mapf.socs.append(soc)
                    stats.mapf.elapsed_times.append(elapsed)

        return stats

    @classmethod
    def load_from_json(cls, filepath: str) -> TrainingStats:
        """Load stats from the legacy single-JSON format."""
        with open(filepath) as f:
            data = json.load(f)

        mapf = None
        if "socs" in data:
            mapf = MAPFStats(
                socs=data["socs"],
                map_size=tuple(data["map_size"]) if data.get("map_size") else None,
                num_agents=data.get("num_agents"),
            )

        losses = data.get("training_loss", data.get("losses", []))
        if not isinstance(losses, list):
            losses = []

        stats = cls(
            training_mode=data.get("training_mode", ""),
            used_device=data.get("used_device"),
            used_seed=data.get("used_seed"),
            mapf=mapf,
        )
        stats._epoch_count = data.get("epochs", len(losses))
        stats._losses = losses
        return stats
