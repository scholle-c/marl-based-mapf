"""Module for storing and handling training statistics."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import json
import numpy as np
import os
from marl_path.constants import TRAIN_MODE_BEST


@dataclass
class TrainingStats:
    """Class for storing training statistics."""

    training_mode: str
    epochs: int = 0
    training_loss: List[float] = field(default_factory=list)
    validation_loss: List[float] = field(default_factory=list)
    socs: List[int] = field(default_factory=list)
    socs_model: List[int | None] = field(default_factory=list)
    socs_no_model: List[int | None] = field(default_factory=list)
    dist_table_differences: List[float | None] = field(default_factory=list)
    dist_table_record_mode: int = 0
    dist_tables_lacam: List = field(default_factory=list)
    dist_tables_model: List = field(default_factory=list)

    @property
    def used_best_mode(self) -> bool:
        """Check if best mode was used"""
        return self.training_mode == TRAIN_MODE_BEST

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
        train_loss: float,
        soc: int,
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

        if self.dist_table_record_mode == 1:
            if (self.epochs - 1) % 10 == 0:
                self.dist_tables_model.append(dist_tables_model)
                if dist_tables_lacam is not None:
                    self.dist_tables_lacam.append(dist_tables_lacam)
        elif self.dist_table_record_mode == 2:
            if self.epochs == 1:
                return
            if (
                self.dist_tables_lacam is not None
                and self.socs_model[-1] is not None
                and self.socs_no_model[-1] is not None
                and self.socs_model[-1] < self.socs_no_model[-1]
            ):
                self.dist_tables_model.append(dist_tables_model)
                self.dist_tables_lacam.append(dist_tables_lacam)
        elif self.dist_table_record_mode == 3:
            self.dist_tables_model.append(dist_tables_model)
            if dist_tables_lacam is not None:
                self.dist_tables_lacam.append(dist_tables_lacam)

    def save(self, output_folder: str) -> None:
        """Save statistics and distance tables to JSON files."""
        filepath_stats = os.path.join(output_folder, "training_stats.json")
        self._save_as_json(filepath_stats)
        if self.dist_table_record_mode != 0:
            self._save_dist_tables(output_folder)

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
            filepath_lacam = os.path.join(output_folder, "dist_tables_lacam.csv")
            np.savetxt(filepath_lacam, dist_tables_lacam_array, delimiter=",", fmt="%d")

        if self.dist_tables_model:
            dist_tables_model_array = concat_dist_tables(self.dist_tables_model)
            filepath_model = os.path.join(output_folder, "dist_tables_model.csv")
            np.savetxt(filepath_model, dist_tables_model_array, delimiter=",", fmt="%d")

    def _save_as_json(self, filepath: str) -> None:
        """Save statistics to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self._to_dict(), f, indent=4)

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
        )

    def _to_dict(self) -> dict[str, List | int | str]:
        """Convert statistics to a dictionary."""
        return {
            "training_mode": self.training_mode,
            "epochs": self.epochs,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "socs": self.socs,
            "socs_model": self.socs_model,
            "socs_no_model": self.socs_no_model,
            "dist_table_differences": self.dist_table_differences,
        }
