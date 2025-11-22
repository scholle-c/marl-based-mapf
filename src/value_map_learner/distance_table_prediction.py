"""Template to train a CNN that predicts MAPF distance tables from value-map inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from value_map_learner.training_data_generation import load_training_data


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
        # x shape: (batch, 3, H, W) -> returns (batch, 1, H, W)
        return self.network(x)


class DistanceTableDataset(Dataset):
    """Wrap saved training data (.npz) so it can be consumed by PyTorch."""

    def __init__(self, archives: Sequence[str | Path]):
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for archive in archives:
            inputs, labels = load_training_data(archive)
            # Convert to tensors once to avoid repeated work during training
            tensor_inputs = torch.from_numpy(inputs).float()
            tensor_labels = torch.from_numpy(labels).unsqueeze(1).float()
            self.samples.extend(zip(tensor_inputs, tensor_labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Implement your preferred loss function (e.g., L1, masked MSE) here."""
    raise NotImplementedError("Define a loss tailored to your training objective.")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
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
    train_archives: Sequence[str | Path]
    val_archives: Sequence[str | Path] | None = None
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_training(config: TrainingConfig) -> nn.Module:
    device = torch.device(config.device)

    train_dataset = DistanceTableDataset(config.train_archives)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_loader = None
    if config.val_archives:
        val_dataset = DistanceTableDataset(config.val_archives)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = DistanceTableCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        log_message = f"Epoch {epoch:03d} | train loss: {train_loss:.4f}"
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            log_message += f" | val loss: {val_loss:.4f}"
        print(log_message)

    return model


def load_archives_from_paths(paths: Iterable[str]) -> list[Path]:
    return [Path(p) for p in paths]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN to predict MAPF distance tables.")
    parser.add_argument("--train-data", nargs="+", required=True, help="Paths to .npz training archives.")
    parser.add_argument("--val-data", nargs="*", help="Optional paths to validation archives.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    config = TrainingConfig(
        train_archives=args.train_data,
        val_archives=args.val_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )
    run_training(config)


if __name__ == "__main__":
    main()
