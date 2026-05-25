import argparse
from pathlib import Path
from marl_path.shared.config import load_config
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()

    # ======== Arguments for PyLaCAM ========
    parser.add_argument(
        "-c",
        "--config-file",
        type=Path,
        default=Path(__file__).parent.parent.parent / "configs" / "default_config.toml",
    )

    parser.add_argument(
        "-m",
        "--map-file",
        type=Path,
        default=Path(__file__).parent.parent.parent / "assets" / "tunnel.map",
    )
    parser.add_argument(
        "-i",
        "--scen-file",
        type=Path,
        default=Path(__file__).parent.parent.parent / "assets" / "tunnel.scen",
    )
    parser.add_argument(
        "-N",
        "--num-agents",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--flg-star",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="choose LaCAM* (default) or vanilla LaCAM",
    )
    parser.add_argument("-s", "--seed", type=int, default=0)

    parser.add_argument("-t", "--time-limit-ms", type=int, default=1000)

    # ======== Arguments for training the distance table predictor ========
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default="vdn",
        help="Choose between: 'vdn': train the heuristic model using VDN loss",
    )

    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="path to a pretrained heuristic model",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="number of training epochs for the heuristic model. Each epoch includes solving an entire mapf instance.",
    )

    parser.add_argument(
        "--model-initialization-mode",
        type=int,
        default=0,
        help="mode for initializing the heuristic model. 0: random initialization, 1: pretrain on default value max map size (= width + height).",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate for training the heuristic model.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to use for training the heuristic model (e.g., 'cpu' or 'cuda').",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="path to save metrics and results of the training and evaluation.",
    )

    parser.add_argument(
        "--seed-training",
        type=int,
        default=0,
        help="random seed for training the heuristic model.",
    )

    # ======== Arguments the visualization afterwards ========
    parser.add_argument(
        "--record-mode",
        type=int,
        default=0,
        help="mode for recording training process. 0: no recording",
    )

    parser.add_argument(
        "--comparison-algorithm",
        type=str,
        default="none",
        help="algorithm for comparison during training. 'none': no comparison, 'lacam': compare with LaCAM, 'model': compare with pretrained model without rl improvements.",
    )

    args = parser.parse_args()
    if args.config_file is not None:
        config = load_config(args.config_file)
        args.__dict__.update(config)

    run_pipeline(args)


if __name__ == "__main__":
    main()
