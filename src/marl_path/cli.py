import argparse
from pathlib import Path

from marl_path.delay_methods import DELAY_METHODS, get_delay_method
from marl_path.path_noise import PATH_NOISE_OPS
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
    parser.add_argument(
        "--cbs-path-penalty",
        type=float,
        default=100000.0,
        help="Penalty added for cells off the time-indexed CBS path "
        "(--delay-method cbs_path). Large = hard constraint, small = hint.",
    )
    parser.add_argument(
        "--path-noise",
        type=str,
        default="none",
        choices=list(PATH_NOISE_OPS),
        help="Structure-preserving noise applied to the CBS paths before the "
        "heuristic is built. Never affects soc_cbs.",
    )
    parser.add_argument(
        "--path-noise-level",
        type=float,
        default=0.0,
        help="Fraction of agents perturbed (0.0-1.0).",
    )
    parser.add_argument(
        "--path-noise-ops",
        type=int,
        default=1,
        help="Perturbations applied per affected agent.",
    )
    parser.add_argument(
        "--path-noise-p",
        type=float,
        default=0.05,
        help="Per-step error probability for --path-noise action (Operator C).",
    )
    parser.add_argument(
        "--path-noise-seed",
        type=int,
        default=0,
        help="RNG seed for noise. Which agents are hit matters a lot — vary "
        "this and report the spread.",
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
