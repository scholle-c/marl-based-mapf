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
class TrainingStats:
    """Class for storing training statistics."""

    training_mode: str
    epochs: int = 0
    training_loss: List[float | None] = field(default_factory=list)
    validation_loss: List[float | None] = field(default_factory=list)
    socs: List[int | None] = field(default_factory=list)
    socs_model: List[int | None] = field(default_factory=list)
    socs_no_model: List[int | None] = field(default_factory=list)
    dist_table_differences: List[float | None] = field(default_factory=list)
    dist_table_record_mode: int = 0
    dist_tables_lacam: List = field(default_factory=list)
    dist_tables_model: List = field(default_factory=list)
    used_device: str | None = None
    used_seed: int | None = None
    map_size: Tuple | None = None
    num_agents: int | None = None
    dist_table_record_granularity: int = 10

    @property
    def used_best_mode(self) -> bool:
        """Check if best mode was used"""
        return self.training_mode == consts.TRAIN_MODE_BEST

    @property
    def has_dist_table_differences(self) -> bool:
        """Check if distance table differences were recorded."""
        return len(self.dist_table_differences) > 0

    @property
    def successful_model_epochs(self) -> int:
        """Count the number of epochs where the model-based solution was better."""
        if not self.used_best_mode:
            return 0
        return sum(
            1
            for soc_model, soc_no_model in zip(self.socs_model, self.socs_no_model)
            if soc_model is not None
            and soc_no_model is not None
            and soc_model < soc_no_model
        )

    def record_epoch(
        self,
        train_loss: float | None,
        soc: int | None,
        val_loss: float | None = None,
        soc_model: int | None = None,
        soc_no_model: int | None = None,
        dist_tables_lacam: List | None = None,
        dist_tables_model: List | None = None,
    ) -> None:
        """Record statistics for a training epoch."""
        self.epochs += 1
        self.training_loss.append(train_loss)
        self.socs.append(soc)

        if val_loss is not None:
            self.validation_loss.append(val_loss)
        if self.used_best_mode:
            self.socs_model.append(soc_model)
            self.socs_no_model.append(soc_no_model)
        if dist_tables_model is not None:
            self._record_dist_table_stats(dist_tables_model, dist_tables_lacam)

    def _record_dist_table_stats(
        self, dist_tables_model: List, dist_tables_lacam: List | None
    ) -> None:
        """Record distance table statistics."""
        if dist_tables_lacam is not None:
            dist_table_difference = float(
                np.mean(
                    [
                        np.abs(dt_model - dt_no_model).mean().item()
                        for dt_model, dt_no_model in zip(
                            dist_tables_model, dist_tables_lacam
                        )
                    ]
                )
            )
            self.dist_table_differences.append(dist_table_difference)

        if (self.epochs - 1) % self.dist_table_record_granularity != 0:
            return

        if self.dist_table_record_mode == 1:
            self.dist_tables_model.append(dist_tables_model)
            self._add_distance_tables_lacam(dist_tables_lacam)
        elif self.dist_table_record_mode == 2:
            self.dist_tables_model.append([dist_tables_model[0]])
            self._add_distance_tables_lacam(dist_tables_lacam, use_one_agent=True)

    def _add_distance_tables_lacam(
        self, dist_tables_lacam: List | None, use_one_agent: bool = False
    ):
        if STORE_LACAM_TABLE_ONLY_ONCE and self.epochs > 1:
            return

        if dist_tables_lacam is not None:
            if use_one_agent:
                self.dist_tables_lacam.append([dist_tables_lacam[0]])
            else:
                self.dist_tables_lacam.append(dist_tables_lacam)

    def save(self, output_folder: str, map_mask: np.ndarray | None = None) -> None:
        """Save statistics and distance tables to JSON files."""
        filepath_stats = os.path.join(
            output_folder, consts.DEFAULT_FILENAME_TRAINING_STATS
        )
        self._save_as_json(filepath_stats)
        if self.dist_table_record_mode != 0:
            self._save_dist_tables(output_folder)
        if map_mask is not None:
            filepath_map = os.path.join(output_folder, consts.DEFAULT_FILENAME_MAP_MASK)
            np.savetxt(filepath_map, map_mask.astype(int), delimiter=",", fmt="%d")

    def _save_dist_tables(self, output_folder: str) -> None:
        """Save distance tables to a csv file."""

        def concat_dist_tables(dist_tables: List[List]) -> np.ndarray:
            """Concatenate distance tables into a single numpy array."""
            all_tables = []

            for tables in dist_tables:
                row = np.hstack(tables)
                all_tables.append(row)
            return np.vstack(all_tables)

        if self.dist_tables_lacam:
            dist_tables_lacam_array = concat_dist_tables(self.dist_tables_lacam)
            filepath_lacam = os.path.join(
                output_folder, consts.DEFAULT_FILENAME_DIST_TABLE_LACAM
            )
            np.savetxt(
                filepath_lacam, dist_tables_lacam_array, delimiter=",", fmt="%.2f"
            )

        if self.dist_tables_model:
            dist_tables_model_array = concat_dist_tables(self.dist_tables_model)
            filepath_model = os.path.join(
                output_folder, consts.DEFAULT_FILENAME_DIST_TABLE_MODEL
            )
            np.savetxt(
                filepath_model, dist_tables_model_array, delimiter=",", fmt="%.2f"
            )

    def _save_as_json(self, filepath: str) -> None:
        """Save statistics to a JSON file."""
        data = self._to_dict()
        data["version"] = __version__
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_json(cls, filepath: str) -> TrainingStats:
        """Load statistics from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(
            training_mode=data["training_mode"],
            epochs=data["epochs"],
            training_loss=data["training_loss"],
            validation_loss=data["validation_loss"],
            socs=data["socs"],
            socs_model=data["socs_model"],
            socs_no_model=data["socs_no_model"],
            dist_table_differences=data["dist_table_differences"],
            map_size=data["map_size"],
            num_agents=data["num_agents"],
        )

    def _to_dict(self) -> dict[str, Any]:
        """Convert statistics to a dictionary."""
        return {
            "training_mode": self.training_mode,
            "used_device": self.used_device,
            "used_seed": self.used_seed,
            "map_size": self.map_size,
            "num_agents": self.num_agents,
            "epochs": self.epochs,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "socs": self.socs,
            "socs_model": self.socs_model,
            "socs_no_model": self.socs_no_model,
            "dist_table_differences": self.dist_table_differences,
        }
