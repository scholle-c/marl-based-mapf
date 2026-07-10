import argparse
from pathlib import Path
from marl_path.shared.config import load_config
from .pipelines.pipeline import run_pipeline
from . import constants as consts


def _dev_path(*parts: str) -> Path | None:
    """Return a path relative to the project root only if it actually exists."""
    candidate = Path(__file__).parent.parent.parent.joinpath(*parts)
    return candidate if candidate.is_file() else None


def _dev_dir(*parts: str) -> Path | None:
    """Return a directory relative to the project root only if it actually exists."""
    candidate = Path(__file__).parent.parent.parent.joinpath(*parts)
    return candidate if candidate.is_dir() else None


def _dev_file(*parts: str) -> Path | None:
    """Return a file relative to the project root only if it actually exists."""
    candidate = Path(__file__).parent.parent.parent.joinpath(*parts)
    return candidate if candidate.is_file() else None


def main():
    parser = argparse.ArgumentParser(
        description="MARL-path: supervised heuristic learning for MAPF."
    )

    parser.add_argument(
        "-c",
        "--config-file",
        type=Path,
        default=_dev_path("configs", "default_config.toml"),
    )

    # ── Pipeline mode ───────────────────────────────────────────────────────
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default=consts.PIPELINE_MODE_SUPERVISED_DELAY,
        choices=[
            consts.PIPELINE_MODE_SUPERVISED_DELAY,
            consts.PIPELINE_MODE_LACAM_ONLY,
        ],
        help=(
            f"'{consts.PIPELINE_MODE_SUPERVISED_DELAY}': train model on CBS-optimal dataset. "
            f"'{consts.PIPELINE_MODE_LACAM_ONLY}': run LaCAM baseline only."
        ),
    )

    # ── Dataset (supervised_delay mode) ────────────────────────────────────
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Directory of .npz files produced by marl-generate. Required for supervised_delay.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of dataset instances used for validation (default: 0.1).",
    )
    parser.add_argument(
        "--delay-target",
        type=str,
        default=consts.DELAY_TARGET_FIRST_VISIT,
        choices=[
            consts.DELAY_TARGET_FIRST_VISIT,
            consts.DELAY_TARGET_NON_OPTIMAL_PENALTY,
        ],
        help=(
            f"'{consts.DELAY_TARGET_FIRST_VISIT}': sparse per-path-cell regression "
            f"target, softplus head, MSE loss (original). "
            f"'{consts.DELAY_TARGET_NON_OPTIMAL_PENALTY}': dense full-grid binary "
            f"mask target, sigmoid head, BCE loss."
        ),
    )
    parser.add_argument(
        "--delay-method",
        type=str,
        default="non_optimal_penalty",
        help=(
            "Delay method name from marl_path.delay_methods.DELAY_METHODS, used "
            f"to build dense targets when --delay-target={consts.DELAY_TARGET_NON_OPTIMAL_PENALTY}."
        ),
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=0.05,
        help=(
            "BCEWithLogitsLoss pos_weight for dense/non_optimal_penalty training. "
            "Down-weights the majority 'off-path' class (label 1); start with "
            "inverse class frequency and tune (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--penalty-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the sigmoid delay output before adding it to h_bfs "
            "at inference (DistTable.compute_delay_model). Only used for sigmoid "
            "output heads (default: 1.0)."
        ),
    )

    # ── MAPF instance (required for lacam_only; optional eval for supervised_delay) ──
    parser.add_argument(
        "-m",
        "--map-file",
        type=Path,
        default=_dev_path("assets", "tunnel.map"),
        help="Path to the .map file. Used for Track B eval in supervised mode and required for lacam_only.",
    )
    parser.add_argument(
        "-i",
        "--scen-file",
        type=Path,
        default=_dev_path("assets", "tunnel.scen"),
        help="Path to the .scen file.",
    )
    parser.add_argument(
        "-N",
        "--num-agents",
        type=int,
        default=4,
    )

    # ── LaCAM solver settings ───────────────────────────────────────────────
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument(
        "--flg-star",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LaCAM* (default) or vanilla LaCAM.",
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-t", "--time-limit-ms", type=int, default=3000)
    parser.add_argument(
        "--cbs-binary",
        type=Path,
        default=_dev_file("CBSH2-RTC", "cbs"),
        help="Path to the CBS (CBSH2-RTC) binary used for optimal-gap reporting.",
    )

    # ── Training hyperparameters ────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of dataset instances per gradient step.",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for training: 'cpu', 'cuda', 'mps', or 'auto'.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Run Track B (LaCAM SOC) eval every N epochs (default: 5).",
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        default=5,
        help="Number of LaCAM runs per Track B eval to average SOC over (default: 5).",
    )

    # ── Model initialisation ────────────────────────────────────────────────
    parser.add_argument(
        "--feature-extractor-type",
        type=str,
        default=consts.EXTRACTOR_BASIC,
        choices=[
            consts.EXTRACTOR_BASIC,
            consts.EXTRACTOR_BINARY_AGENTS_CHANNEL,
            consts.EXTRACTOR_AGGREGATED_AGENTS_CHANNEL,
            consts.EXTRACTOR_RICH_AGENTS_CHANNEL,
        ],
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
        help="Conv channel width of DistanceTableCNN (default: 32).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of conv blocks in DistanceTableCNN (default: 4).",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="Path to a pretrained model checkpoint to continue training from.",
    )
    parser.add_argument(
        "--model-initialization-mode",
        type=int,
        default=0,
        help="0: random init, 1: pretrain on constant, 2: pretrain on BFS.",
    )
    parser.add_argument("--seed-training", type=int, default=0)

    # ── Output ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "default_output",
    )
    parser.add_argument(
        "--record-mode",
        type=int,
        default=1,
        help="0: no output, 1: save model + metrics.",
    )

    # Load config file defaults before final parse.
    first_pass, _ = parser.parse_known_args()
    if first_pass.config_file is not None:
        config = load_config(first_pass.config_file)
        parser.set_defaults(**config)

    args = parser.parse_args()

    # ── Validate mode-specific requirements ────────────────────────────────
    if args.pipeline_mode == consts.PIPELINE_MODE_SUPERVISED_DELAY:
        if args.dataset_dir is None:
            parser.error("--dataset-dir is required for supervised_delay mode.")
    elif args.pipeline_mode == consts.PIPELINE_MODE_LACAM_ONLY:
        if args.map_file is None or args.scen_file is None:
            parser.error("--map-file and --scen-file are required for lacam_only mode.")

    run_pipeline(args)


if __name__ == "__main__":
    main()
