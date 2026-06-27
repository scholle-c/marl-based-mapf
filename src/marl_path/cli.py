import argparse
from pathlib import Path

from marl_path.delay_methods import DELAY_METHODS, get_delay_method
from marl_path.pipeline import run_evaluation
from marl_path.shared.config import load_config


def _dev_path(*parts: str) -> Path | None:
    candidate = Path(__file__).parent.parent.parent.joinpath(*parts)
    return candidate if candidate.is_file() else None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LaCAM with a precomputed delay table against the plain baseline."
    )

    parser.add_argument(
        "-c",
        "--config-file",
        type=Path,
        default=_dev_path("configs", "default_config.toml"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory of .npz files produced by marl-generate.",
    )
    parser.add_argument(
        "--delay-method",
        type=str,
        default="first_visit",
        choices=list(DELAY_METHODS),
        help="Method used to compute per-agent delay maps from CBS paths (default: first_visit).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-t",
        "--time-limit-ms",
        type=int,
        default=3000,
    )
    parser.add_argument(
        "--flg-star",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LaCAM* (default) or vanilla LaCAM.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write results JSON.",
    )

    first_pass, _ = parser.parse_known_args()
    if first_pass.config_file is not None and first_pass.config_file.exists():
        config = load_config(first_pass.config_file)
        parser.set_defaults(**config)

    args = parser.parse_args()
    delay_method = get_delay_method(args.delay_method)
    run_evaluation(args, delay_method)


if __name__ == "__main__":
    main()
