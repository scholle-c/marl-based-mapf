"""Module for storing and handling training statistics."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Any, Tuple
import json
import numpy as np
import os
import marl_path.constants as consts
from importlib.metadata import version

__version__ = version("marl-path")

STORE_LACAM_TABLE_ONLY_ONCE: bool = True


@dataclass
class LearningStats:
    """Tracks training and validation losses."""

    training_loss: List[float | None] = field(default_factory=list)
    validation_loss: List[float | None] = field(default_factory=list)


@dataclass
class MAPFStats:
    """Tracks MAPF solution quality metrics."""

    socs: List[int | None] = field(default_factory=list)
    socs_model: List[int | None] = field(default_factory=list)
    socs_no_model: List[int | None] = field(default_factory=list)
    num_agents: int | None = None
    map_size: Tuple | None = None


@dataclass
class DistTableStats:
    """Tracks distance table comparison metrics and optionally full tables.

    record_mode: 0 = differences only, 1 = all agents, 2 = one agent
    """

    dist_table_differences: List[float | None] = field(default_factory=list)
    dist_tables_lacam: List = field(default_factory=list)
    dist_tables_model: List = field(default_factory=list)
    record_mode: int = 0
    record_granularity: int = 10

    def record(
        self, epochs: int, dist_tables_model: List, dist_tables_lacam: List | None
    ) -> None:
        """Record distance table stats for the current epoch."""
        if dist_tables_lacam is not None:
            diff = float(
                np.mean(
                    [
                        np.abs(dt_model - dt_no_model).mean().item()
                        for dt_model, dt_no_model in zip(
                            dist_tables_model, dist_tables_lacam
                        )
                    ]
                )
            )
            self.dist_table_differences.append(diff)

        if (epochs - 1) % self.record_granularity != 0:
            return

        if self.record_mode == 1:
            self.dist_tables_model.append(dist_tables_model)
            self._add_lacam_tables(epochs, dist_tables_lacam)
        elif self.record_mode == 2:
            self.dist_tables_model.append([dist_tables_model[0]])
            self._add_lacam_tables(epochs, dist_tables_lacam, use_one_agent=True)

    def _add_lacam_tables(
        self, epochs: int, dist_tables_lacam: List | None, use_one_agent: bool = False
    ) -> None:
        if STORE_LACAM_TABLE_ONLY_ONCE and epochs > 1:
            return
        if dist_tables_lacam is not None:
            if use_one_agent:
                self.dist_tables_lacam.append([dist_tables_lacam[0]])
            else:
                self.dist_tables_lacam.append(dist_tables_lacam)

    def save(self, output_folder: str) -> None:
        """Save full distance tables to CSV files."""

        def concat_dist_tables(dist_tables: List[List]) -> np.ndarray:
            all_tables = []
            for tables in dist_tables:
                all_tables.append(np.hstack(tables))
            return np.vstack(all_tables)

        if self.dist_tables_lacam:
            arr = concat_dist_tables(self.dist_tables_lacam)
            np.savetxt(
                os.path.join(output_folder, consts.DEFAULT_FILENAME_DIST_TABLE_LACAM),
                arr,
                delimiter=",",
                fmt="%.2f",
            )

        if self.dist_tables_model:
            arr = concat_dist_tables(self.dist_tables_model)
            np.savetxt(
                os.path.join(output_folder, consts.DEFAULT_FILENAME_DIST_TABLE_MODEL),
                arr,
                delimiter=",",
                fmt="%.2f",
            )


@dataclass
class TrainingStats:
    """Stores training statistics. Compose with optional sub-stats for MAPF and dist table tracking."""

    training_mode: str
    epochs: int = 0
    learning: LearningStats = field(default_factory=LearningStats)
    mapf: MAPFStats | None = None
    dist_tables: DistTableStats | None = None
    used_device: str | None = None
    used_seed: int | None = None

    @property
    def used_best_mode(self) -> bool:
        return self.training_mode == consts.TRAIN_MODE_BEST

    @property
    def has_dist_table_differences(self) -> bool:
        return (
            self.dist_tables is not None
            and len(self.dist_tables.dist_table_differences) > 0
        )

    @property
    def successful_model_epochs(self) -> int:
        """Count epochs where the model-based solution was better than LaCAM."""
        if not self.used_best_mode or self.mapf is None:
            return 0
        return sum(
            1
            for soc_model, soc_no_model in zip(
                self.mapf.socs_model, self.mapf.socs_no_model
            )
            if soc_model is not None
            and soc_no_model is not None
            and soc_model < soc_no_model
        )

    def record_epoch(
        self,
        train_loss: float | None,
        soc: int | None = None,
        val_loss: float | None = None,
        soc_model: int | None = None,
        soc_no_model: int | None = None,
        dist_tables_lacam: List | None = None,
        dist_tables_model: List | None = None,
    ) -> None:
        """Record statistics for a training epoch."""
        self.epochs += 1
        self.learning.training_loss.append(train_loss)
        if val_loss is not None:
            self.learning.validation_loss.append(val_loss)

        if self.mapf is not None:
            self.mapf.socs.append(soc)
            if self.used_best_mode:
                self.mapf.socs_model.append(soc_model)
                self.mapf.socs_no_model.append(soc_no_model)

        if self.dist_tables is not None and dist_tables_model is not None:
            self.dist_tables.record(self.epochs, dist_tables_model, dist_tables_lacam)

    def save(self, output_folder: str, map_mask: np.ndarray | None = None) -> None:
        """Save statistics and distance tables to files."""
        self._save_as_json(
            os.path.join(output_folder, consts.DEFAULT_FILENAME_TRAINING_STATS)
        )
        if self.dist_tables is not None and self.dist_tables.record_mode != 0:
            self.dist_tables.save(output_folder)
        if map_mask is not None:
            np.savetxt(
                os.path.join(output_folder, consts.DEFAULT_FILENAME_MAP_MASK),
                map_mask.astype(int),
                delimiter=",",
                fmt="%d",
            )

    def _save_as_json(self, filepath: str) -> None:
        data = self._to_dict()
        data["version"] = __version__
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def _to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "training_mode": self.training_mode,
            "used_device": self.used_device,
            "used_seed": self.used_seed,
            "epochs": self.epochs,
            "training_loss": self.learning.training_loss,
            "validation_loss": self.learning.validation_loss,
        }
        if self.mapf is not None:
            d["map_size"] = self.mapf.map_size
            d["num_agents"] = self.mapf.num_agents
            d["socs"] = self.mapf.socs
            d["socs_model"] = self.mapf.socs_model
            d["socs_no_model"] = self.mapf.socs_no_model
        if self.dist_tables is not None:
            d["dist_table_differences"] = self.dist_tables.dist_table_differences
        return d

    @classmethod
    def load_from_json(cls, filepath: str) -> TrainingStats:
        """Load statistics from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        mapf = None
        if "socs" in data:
            mapf = MAPFStats(
                socs=data["socs"],
                socs_model=data.get("socs_model", []),
                socs_no_model=data.get("socs_no_model", []),
                map_size=tuple(data["map_size"]) if data.get("map_size") else None,
                num_agents=data.get("num_agents"),
            )

        dist_tables = None
        if "dist_table_differences" in data:
            dist_tables = DistTableStats(
                dist_table_differences=data["dist_table_differences"],
            )

        return cls(
            training_mode=data["training_mode"],
            epochs=data["epochs"],
            learning=LearningStats(
                training_loss=data["training_loss"],
                validation_loss=data.get("validation_loss", []),
            ),
            mapf=mapf,
            dist_tables=dist_tables,
            used_device=data.get("used_device"),
            used_seed=data.get("used_seed"),
        )
