"""Template to train a CNN that predicts MAPF distance tables from value-map inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import json

from .training_data_generation import load_training_data

class DistanceTableCNN(nn.Module):
    """
    Plain CNN without pooling that maps 3-channel inputs (map, goal, start)
    to a single-channel distance map.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(depth - 1):
            layers.append(nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            channels = hidden_channels
        layers.append(nn.Conv2d(channels, 1, kernel_size=3, padding=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, 3, H, W) -> (batch, 1, H, W)."""
        return self.network(x)


class DistanceTableDataset(Dataset):
    """Wrap saved or in-memory training data so it can be consumed by PyTorch."""

    def __init__(
        self,
        archives: Sequence[str | Path] | None = None,
        in_memory_data: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        if archives:
            for archive in archives:
                inputs, labels = load_training_data(archive)
                # Convert to tensors once to avoid repeated work during training
                tensor_inputs = torch.from_numpy(inputs).float()
                tensor_labels = torch.from_numpy(labels).unsqueeze(1).float()
                self.samples.extend(zip(tensor_inputs, tensor_labels))

        if in_memory_data:
            for inputs, labels in in_memory_data:
                tensor_inputs = torch.from_numpy(inputs).float()
                tensor_labels = torch.from_numpy(labels).unsqueeze(1).float()
                self.samples.extend(zip(tensor_inputs, tensor_labels))

        if not self.samples:
            raise ValueError("No training samples provided to DistanceTableDataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Masked MSE that ignores cells marked as unreachable (sentinel = map size).

    The target tensors use the map size (height * width) as a sentinel value
    for unreachable cells. Those positions are excluded from the loss.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Predictions shape {predictions.shape} must match targets {targets.shape}.")

    # Sentinel is the maximum value per sample (grid.size). Keep dimensions for broadcasting.
    sentinel = targets.amax(dim=(2, 3), keepdim=True)
    mask = targets != sentinel

    if not mask.any():
        raise ValueError("All target values are sentinel values; nothing to train on.")

    squared_error = (predictions - targets) ** 2
    masked_error = squared_error[mask]
    return masked_error.mean()


def pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that pads variable-sized maps to the largest H/W in the batch.

    Inputs are padded with zeros; labels are padded with the sample's sentinel
    (max value) so the masked loss will ignore the padded region.
    """
    inputs_list, labels_list = zip(*batch)
    max_h = max(item.shape[-2] for item in inputs_list)
    max_w = max(item.shape[-1] for item in inputs_list)

    padded_inputs: list[torch.Tensor] = []
    padded_labels: list[torch.Tensor] = []

    for x, y in zip(inputs_list, labels_list):
        pad_h = max_h - x.shape[-2]
        pad_w = max_w - x.shape[-1]
        # Pad format: (left, right, top, bottom) for 2D spatial dims
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        sentinel = y.max()
        y_padded = F.pad(y, (0, pad_w, 0, pad_h), value=sentinel)
        padded_inputs.append(x_padded)
        padded_labels.append(y_padded)

    return torch.stack(padded_inputs), torch.stack(padded_labels)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return mean loss over the dataset."""
    model.train()
    epoch_loss = 0.0
    for batch_inputs, batch_labels in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        predictions = model(batch_inputs)
        loss = compute_loss(predictions, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_inputs.size(0)

    return epoch_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate the model on a dataloader and return mean loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            predictions = model(batch_inputs)
            loss = compute_loss(predictions, batch_labels)
            total_loss += loss.item() * batch_inputs.size(0)
    return total_loss / len(dataloader.dataset)


@dataclass
class TrainingConfig:
    train_archives: Sequence[str | Path] | None = None
    train_data: Sequence[tuple[np.ndarray, np.ndarray]] | None = None
    val_archives: Sequence[str | Path] | None = None
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: Path | str = Path("output")


def run_training(config: TrainingConfig) -> nn.Module:
    """Train the CNN with provided data sources and persist artifacts."""
    device = torch.device(config.device)

    if not config.train_archives and not config.train_data:
        raise ValueError("Provide either train_archives or in-memory train_data.")

    train_dataset = DistanceTableDataset(
        archives=config.train_archives,
        in_memory_data=config.train_data,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
    )

    val_loader = None
    if config.val_archives:
        val_dataset = DistanceTableDataset(config.val_archives)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=pad_collate,
        )

    model = DistanceTableCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: dict[str, list[float]] = {"train": [], "val": []}

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        history["train"].append(train_loss)
        log_message = f"Epoch {epoch:03d} | train loss: {train_loss:.4f}"
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            history["val"].append(val_loss)
            log_message += f" | val loss: {val_loss:.4f}"
        print(log_message)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f)
    return model


def load_archives_from_paths(paths: Iterable[str]) -> list[Path]:
    """Convert iterable of string paths to Path objects."""
    return [Path(p) for p in paths]
