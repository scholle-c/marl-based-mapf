"""Command-line interface for data generation and training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .config import load_config
from .distance_table_prediction import TrainingConfig, run_training
from .training_data_generation import create_training_data, save_training_data


def _save_generated_data(datasets, output_dir: str, map_paths: Sequence[str]) -> list[Path]:
    """Persist generated datasets to disk, using map stems for filenames."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    if len(datasets) != len(map_paths):
        raise ValueError("Number of datasets does not match number of map paths.")

    name_counts: dict[str, int] = {}
    for idx, (map_path, (inputs, labels)) in enumerate(zip(map_paths, datasets)):
        stem = Path(map_path).stem or f"map_{idx:03d}"
        suffix_idx = name_counts.get(stem, 0)
        name_counts[stem] = suffix_idx + 1
        suffix = f"_{suffix_idx}" if suffix_idx else ""
        archive_path = output_dir_path / f"training_data_{stem}{suffix}.npz"
        save_training_data(inputs, labels, archive_path)
        saved_paths.append(archive_path)
    return saved_paths


def _pick(cli_value, config_value, default=None):
    """Return CLI value if provided, else config value, else default."""
    return cli_value if cli_value is not None else config_value if config_value is not None else default


def _load_config_data(config_path: str | None) -> dict[str, Any]:
    """Load configuration from a file path or return an empty dict."""
    if not config_path:
        return {}
    return load_config(config_path) or {}


def _resolve_generation_args(
    args: argparse.Namespace, config_data: dict[str, Any]
) -> tuple[list[str] | None, int | None, str | None]:
    """Resolve generation-specific arguments from CLI/config."""
    maps = _pick(args.maps, config_data.get("maps"))
    samples_per_map = _pick(args.samples_per_map, config_data.get("samples_per_map"))
    output_dir = _pick(args.output_dir, config_data.get("output_dir"))
    return maps, samples_per_map, output_dir


def _resolve_training_args(
    args: argparse.Namespace, config_data: dict[str, Any], defaults: TrainingConfig
) -> dict[str, Any]:
    """Resolve training-related parameters from CLI/config with defaults."""
    return {
        "train_archives": _pick(args.train_data, config_data.get("train_archives")),
        "val_archives": _pick(args.val_data, config_data.get("val_archives")),
        "maps": _pick(args.maps, config_data.get("maps")),
        "samples_per_map": _pick(args.samples_per_map, config_data.get("samples_per_map")),
        "save_generated_data": _pick(args.save_generated_data, config_data.get("save_generated_data")),
        "batch_size": _pick(args.batch_size, config_data.get("batch_size"), defaults.batch_size),
        "epochs": _pick(args.epochs, config_data.get("epochs"), defaults.epochs),
        "learning_rate": _pick(args.lr, config_data.get("learning_rate"), defaults.learning_rate),
        "device": _pick(args.device, config_data.get("device"), defaults.device),
        "output_dir": _pick(args.output_dir, config_data.get("output_dir"), defaults.output_dir),
    }


def _build_training_sources(
    maps: Sequence[str] | None, samples_per_map: int | None, save_generated_data: str | None
) -> tuple[Sequence[str] | None, Sequence[tuple[Any, Any]] | None]:
    """Generate datasets from maps, optionally persisting them, and return archives or in-memory data."""
    if not maps:
        return None, None
    datasets = create_training_data(maps, samples_per_map)
    if save_generated_data:
        saved_paths = _save_generated_data(datasets, save_generated_data, maps)
        return [str(path) for path in saved_paths], None
    return None, datasets


def _configure_train_parser(subparsers) -> None:
    """Attach the train subcommand and its arguments."""
    train_parser = subparsers.add_parser("train", help="Train model from archives or generated data.")
    train_parser.add_argument("--config", type=str, help="Path to JSON/TOML/YAML config file.")
    train_parser.add_argument("--train-data", nargs="+", help="Paths to .npz training archives.")
    train_parser.add_argument("--val-data", nargs="*", help="Optional paths to validation archives.")
    train_parser.add_argument("--maps", nargs="+", help="Paths to map files to generate training data.")
    train_parser.add_argument(
        "--samples-per-map",
        type=int,
        help="Number of samples to generate per map when using --maps.",
    )
    train_parser.add_argument(
        "--save-generated-data",
        type=str,
        help="Directory to store generated training data archives (.npz).",
    )
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--lr", type=float)
    train_parser.add_argument("--device", type=str)
    train_parser.add_argument("--output-dir", type=str, help="Directory to store model checkpoints and loss history.")
    train_parser.set_defaults(func=_cmd_train)


def _configure_generate_parser(subparsers) -> None:
    """Attach the generate subcommand and its arguments."""
    gen_parser = subparsers.add_parser("generate", help="Generate training data archives from map files.")
    gen_parser.add_argument("--config", type=str, help="Path to JSON/TOML/YAML config file.")
    gen_parser.add_argument("--maps", nargs="+", help="Paths to map files.")
    gen_parser.add_argument(
        "--samples-per-map",
        type=int,
        help="Number of samples to generate per map; defaults to all accessible positions.",
    )
    gen_parser.add_argument(
        "--output-dir",
        help="Directory to store generated training data archives (.npz).",
    )
    gen_parser.set_defaults(func=_cmd_generate)


def _cmd_generate(args: argparse.Namespace) -> None:
    """Handle the `generate` subcommand: produce archives from map files."""
    config_data = _load_config_data(args.config)
    maps, samples_per_map, output_dir = _resolve_generation_args(args, config_data)
    if not maps:
        raise SystemExit("Provide --maps or set maps in the config.")
    if not output_dir:
        raise SystemExit("Provide --output-dir or set output_dir in the config.")

    datasets = create_training_data(maps, samples_per_map)
    saved_paths = _save_generated_data(datasets, output_dir, maps)
    print(f"Generated {sum(inputs.shape[0] for inputs, _ in datasets)} samples across {len(datasets)} map(s).")
    print(f"Saved archives to {Path(output_dir).resolve()}")


def _build_training_config(
    params: dict[str, Any], train_archives: Sequence[str] | None, in_memory_train_data: Sequence[tuple[Any, Any]] | None
) -> TrainingConfig:
    """Create a TrainingConfig from resolved parameters and data sources."""
    return TrainingConfig(
        train_archives=train_archives or None,
        train_data=in_memory_train_data or None,
        val_archives=params["val_archives"],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        learning_rate=params["learning_rate"],
        device=params["device"],
        output_dir=params["output_dir"],
    )


def _cmd_train(args: argparse.Namespace) -> None:
    """Handle the `train` subcommand: resolve inputs, optionally generate data, and train."""
    config_data = _load_config_data(args.config)
    defaults = TrainingConfig()
    params = _resolve_training_args(args, config_data, defaults)

    train_archives = list(params["train_archives"] or [])
    generated_archives, in_memory_train_data = _build_training_sources(
        params["maps"], params["samples_per_map"], params["save_generated_data"]
    )
    if generated_archives:
        train_archives.extend(generated_archives)

    if not train_archives and not in_memory_train_data:
        raise SystemExit("Provide --train-data archives or --maps to generate training data.")

    run_training(_build_training_config(params, train_archives, in_memory_train_data))


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser with subcommands."""
    parser = argparse.ArgumentParser(prog="value-map-learner", description="Value Map Learner CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _configure_generate_parser(subparsers)
    _configure_train_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
