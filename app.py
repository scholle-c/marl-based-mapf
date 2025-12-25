import argparse
from pathlib import Path
from pathfinding_model_pretraining import load_config
from MARL_MAPF_pipeline.pipeline import run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config-file",
        type=Path,
        default=Path(__file__).parent / "configs" / "default_config.toml",
    )

    parser.add_argument(
        "-m",
        "--map-file",
        type=Path,
        default=Path(__file__).parent / "assets" / "tunnel.map",
    )
    parser.add_argument(
        "-i",
        "--scen-file",
        type=Path,
        default=Path(__file__).parent / "assets" / "tunnel.scen",
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
        "--flg_star",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="choose LaCAM* (default) or vanilla LaCAM",
    )
    parser.add_argument("-s", "--seed", type=int, default=0)

    parser.add_argument("-t", "--time_limit_ms", type=int, default=1000)

    parser.add_argument(
        "--training_mode",
        type=str,
        default="model",
        help="Choose between: 'model' (train with model-based solutions), "
        "'lacam_only' (no training, just one LaCAM execution), and 'best' "
        "(take best solution from either LaCAM or model per epoch). Default: 'model'",
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
        "--output-folder",
        type=Path,
        default=Path(__file__).parent / "output",
        help="path to save metrics and results of the training and evaluation.",
    )

    parser.add_argument(
        "--seed_training",
        type=int,
        default=0,
        help="random seed for training the distance table CNN model.",
    )


    args = parser.parse_args()
    if args.config_file is not None:
        config = load_config(args.config_file)
        args.__dict__.update(config)

    run_pipeline(args)
