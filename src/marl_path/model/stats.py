"""Module for storing and handling training statistics."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import json


@dataclass
class TrainingStats:
    """Class for storing training statistics."""

    epochs: int = 0
    training_loss: List[float] = field(default_factory=list)
    validation_loss: List[float] = field(default_factory=list)
    socs: List[int] = field(default_factory=list)
    socs_model: List[int] = field(default_factory=list)
    socs_no_model: List[int] = field(default_factory=list)
    dist_table_differences: List[float] = field(default_factory=list)

    @property
    def used_best_mode(self) -> bool:
        """Check if best mode was used (i.e., if socs_no_model has data)."""
        return len(self.socs_no_model) > 0

    @property
    def has_dist_table_differences(self) -> bool:
        """Check if distance table differences were recorded."""
        return len(self.dist_table_differences) > 0

    def record_epoch(
        self,
        train_loss: float,
        soc: int,
        val_loss: float | None = None,
        soc_model: int | None = None,
        soc_no_model: int | None = None,
        dist_table_diff: float | None = None,
    ) -> None:
        """Record statistics for a training epoch."""
        self.epochs += 1
        self.training_loss.append(train_loss)
        self.socs.append(soc)
        if val_loss is not None:
            self.validation_loss.append(val_loss)
        if soc_model is not None:
            self.socs_model.append(soc_model)
        if soc_no_model is not None:
            self.socs_no_model.append(soc_no_model)
        if dist_table_diff is not None:
            self.dist_table_differences.append(dist_table_diff)

    def save_as_json(self, filepath: str) -> None:
        """Save statistics to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self._to_dict(), f, indent=4)

    @classmethod
    def load_from_json(cls, filepath: str) -> TrainingStats:
        """Load statistics from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(
            epochs=data["epochs"],
            training_loss=data["training_loss"],
            validation_loss=data["validation_loss"],
            socs=data["socs"],
            socs_model=data["socs_model"],
            socs_no_model=data["socs_no_model"],
            dist_table_differences=data["dist_table_differences"],
        )

    def _to_dict(self) -> dict[str, List | int]:
        """Convert statistics to a dictionary."""
        return {
            "epochs": self.epochs,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "socs": self.socs,
            "socs_model": self.socs_model,
            "socs_no_model": self.socs_no_model,
            "dist_table_differences": self.dist_table_differences,
        }
