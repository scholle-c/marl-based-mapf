import argparse
from pathlib import Path

import torch

from value_map_learner import (
    TrainingConfig,
    create_training_data,
    run_training,
    save_training_data,
)


def _save_generated_data(datasets, output_dir: str) -> list[Path]:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for idx, (inputs, labels) in enumerate(datasets):
        archive_path = output_dir_path / f"training_data_{idx:03d}.npz"
        save_training_data(inputs, labels, archive_path)
        saved_paths.append(archive_path)
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN to predict MAPF distance tables.")
    parser.add_argument("--train-data", nargs="+", help="Paths to .npz training archives.")
    parser.add_argument("--val-data", nargs="*", help="Optional paths to validation archives.")
    parser.add_argument(
        "--maps",
        nargs="+",
        help="Paths to map files. Training data is generated from these maps.",
    )
    parser.add_argument(
        "--save-generated-data",
        type=str,
        help="Directory to store generated training data archives (.npz).",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate training data (from --maps) and exit without training.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to store model checkpoints and loss history.",
    )

    args = parser.parse_args()

    train_archives = list(args.train_data) if args.train_data else []
    in_memory_train_data = []

    if args.maps:
        generated_datasets = create_training_data(args.maps)

        if args.save_generated_data:
            saved_paths = _save_generated_data(generated_datasets, args.save_generated_data)
            if not args.generate_only:
                train_archives.extend(str(path) for path in saved_paths)
        elif not args.generate_only:
            in_memory_train_data = generated_datasets

        if args.generate_only:
            print(f"Generated training data for {len(generated_datasets)} maps.")
            if args.save_generated_data:
                print(f"Saved generated archives to {Path(args.save_generated_data).resolve()}")
            else:
                print("Generated data kept in memory only; no training run.")
            return

    if not train_archives and not in_memory_train_data:
        parser.error("Provide --train-data archives or --maps to generate training data.")

    config = TrainingConfig(
        train_archives=train_archives or None,
        train_data=in_memory_train_data or None,
        val_archives=args.val_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        output_dir=args.output_dir,
    )
    run_training(config)


if __name__ == "__main__":
    main()
