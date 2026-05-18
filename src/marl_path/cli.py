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
        "--training-mode",
        type=str,
        default="model",
        help="Choose between: 'model' (train with model-based solutions), "
        "'lacam_only' (no training, just one LaCAM execution), and 'best' "
        "(take best solution from either LaCAM or model per epoch). Default: 'model'",
    )

    parser.add_argument(
        "--epsilon-function",
        type=str,
        default="none",
        help="Is used for exploration: Epsilon is a chance, of choosing the worse solution over the better one to explore alternative solutions. Is only used, when training-mode=best. Choose between: 'none' (no epsilon-greedy), 'sine' (sine function over epochs), 'fixed' (1/4 epochs exploitation, 1/2 exploration, 1/4 exploitation). Default: 'none'",
    )

    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.1,
        help="minimum epsilon value for epsilon-greedy exploration. Only relevant if epsilon-function is not 'none'.",
    )

    parser.add_argument(
        "--epsilon-max",
        type=float,
        default=0.8,
        help="maximum epsilon value for epsilon-greedy exploration. Only relevant if epsilon-function is not 'none'.",
    )

    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="path to a pretrained distance table CNN model",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of training epochs for the distance table CNN model. Each epoch goes over an entire run of the LaCAM planner.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate for training the distance table CNN model.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to use for training the distance table CNN model (e.g., 'cpu' or 'cuda').",
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
        help="random seed for training the distance table CNN model.",
    )

    parser.add_argument(
        "--use-pretraining",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to pretrain the distance table model on the map size as default values before training with LaCAM.",
    )

    parser.add_argument(
        "--use-neighbors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to include neighboring cells in the loss computation when training the distance table model.",
    )

    parser.add_argument(
        "--goal-weight",
        type=float,
        default=1.0,
        help="Weight of the loss value for the goal prediction, which has the target value 0.",
    )

    parser.add_argument(
        "--use-bellman-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to include the Bellman loss when training the distance table model.",
    )

    # ======== Arguments the visualization afterwards ========
    parser.add_argument(
        "--dist-table-record-mode",
        type=int,
        default=0,
        help="mode for recording distance tables during training. 0: no recording, 1: record all epochs (WARINING: storage intensive with large maps/number of agents) 2: record for only one agent",
    )

    parser.add_argument(
        "--dist-table-record-granularity",
        type=int,
        default=10,
        help="granularity for recording distance tables (number of epochs when recording occurs). Only relevant if dist-table-record-mode is not 0.",
    )

    parser.add_argument(
        "--agent-path-record-mode",
        type=int,
        default=0,
        help="mode for recording agent paths during training. 0: no recording, 1: record all epochs (WARINING: storage intensive with large maps/number of agents) 2: record for only one agent",
    )

    parser.add_argument(
        "--agent-path-record-granularity",
        type=int,
        default=10,
        help="granularity for recording agent paths (number of epochs when recording occurs). Only relevant if agent-path-record-mode is not 0.",
    )

    parser.add_argument(
        "--save-map-mask",
        type=bool,
        default=True,
        help="if set to true, an outline of the map is stored for later visualization.",
    )

    args = parser.parse_args()
    if args.config_file is not None:
        config = load_config(args.config_file)
        args.__dict__.update(config)

    run_pipeline(args)


if __name__ == "__main__":
    main()
