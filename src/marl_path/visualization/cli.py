import argparse
import sys
from pathlib import Path
import subprocess
from marl_path.visualization.plotting import run_plotting


def main():
    parser = argparse.ArgumentParser(
        description="Plot training statistics for distance table CNN model."
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help=(
            "Path(s) to JSON stats files or folders containing training_stats.json. "
            "Provide multiple to visualize distribution."
        ),
    )
    args = parser.parse_args()
    if args.stats_file is not None:
        run_plotting(args)
    else:
        app_path = Path(__file__).with_name("app.py")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)], check=True
        )


if __name__ == "__main__":
    main()
