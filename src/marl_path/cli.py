import argparse
from pathlib import Path
from marl_path.shared.config import load_config
from .pipelines.pipeline import run_pipeline
from . import constants as consts


def _dev_path(*parts: str) -> Path | None:
    """Return a path relative to the project root only if it actually exists.

    Works in an editable/dev install where the source tree is present next to
    the package.  Returns None when installed as a regular package (e.g. inside
    a venv on a server) so that argparse falls back to requiring the user to
    supply the value explicitly.
    """
    candidate = Path(__file__).parent.parent.parent.joinpath(*parts)
    return candidate if candidate.is_file() else None


def main():
    parser = argparse.ArgumentParser()

    # ======== Arguments for PyLaCAM ========
    parser.add_argument(
        "-c",
        "--config-file",
        type=Path,
        default=_dev_path("configs", "default_config.toml"),
    )

    parser.add_argument(
        "-m",
        "--map-file",
        type=Path,
        default=_dev_path("assets", "tunnel.map"),
    )

    parser.add_argument(
        "-i",
        "--scen-file",
        type=Path,
        default=_dev_path("assets", "tunnel.scen"),
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
        help=f"Choose between: '{consts.PIPELINE_MODE_VDN}': train the heuristic model using VDN loss. '{consts.PIPELINE_MODE_EXPERT_PRETRAIN}': pretrain the heuristic model using solutions from an expert algorithm. '{consts.PIPELINE_MODE_LACAM_ONLY}': run LaCAM once without any rl training or using a distance table CNN model.",
    )

    parser.add_argument(
        "--training-mode",
        type=str,
        default=consts.TRAINING_MODE_VDN,
        help=f"Tensor/target computation strategy. '{consts.TRAINING_MODE_VDN}': VDN decomposition (sum agent values and targets). '{consts.TRAINING_MODE_INDIVIDUAL}': individual agent path loss (concatenate per-agent values and targets).",
    )

    parser.add_argument(
        "--feature-extractor-type",
        type=str,
        default=consts.EXTRACTOR_BASIC,
        help=f"type of feature extractor to use for the heuristic model. Choose between: {consts.EXTRACTOR_BASIC}: 3 channels (map, goal, start), {consts.EXTRACTOR_OTHER_AGENTS_CHANNEL}: basic + one binary channel marking all other agent positions",
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
        default=10,
        help="number of training epochs for the heuristic model. Each epoch includes the solutions of the size of the batch.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="batch size for training the heuristic model. Each batch includes multiple (state, target) pairs collected from solving mapf instances.",
    )

    parser.add_argument(
        "--model-initialization-mode",
        type=int,
        default=0,
        help="mode for initializing the heuristic model. 0: random initialization, 1: pretrain on default value max map size (= width + height), 2: pretrain on BFS distance tables",
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
        default=Path("output") / "default_output",
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
        default=1,
        help="mode for recording training process. 0: no recording",
    )

    parser.add_argument(
        "--record-num-agents",
        type=int,
        default=0,
        help="number of agents to record during training. 0: record all agents",
    )

    parser.add_argument(
        "--record-paths",
        type=bool,
        default=False,
        help="whether to record the paths of agents during training. 0: do not record, 1: record paths",
    )

    parser.add_argument(
        "--record-heuristics",
        type=bool,
        default=False,
        help="whether to record the heuristics of agents during training. 0: do not record, 1: record heuristics",
    )

    parser.add_argument(
        "--record-episode-interval",
        type=int,
        default=100,
        help="interval (in episodes) at which to record training metrics and results.",
    )

    parser.add_argument(
        "--record-logs",
        type=bool,
        default=True,
        help="whether to record the logs of training. 0: do not record, 1: record logs",
    )

    # Load config as defaults so explicit CLI args can still override them.
    first_pass, _ = parser.parse_known_args()
    if first_pass.config_file is not None:
        config = load_config(first_pass.config_file)
        parser.set_defaults(**config)

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
