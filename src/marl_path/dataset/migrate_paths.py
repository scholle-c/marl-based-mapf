"""Rewrite absolute map_file/scen_file paths in cached .npz instances to be
repo-relative, so datasets generated before this fix can be copied to another
machine (e.g. a training server) without breaking.

Usage:
    marl-migrate-paths data/warehouse-20agents-100sub
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from marl_path.shared.mapf_utils import to_portable_path

from .instance import CachedInstance


def migrate(dataset_dir: Path, dry_run: bool = False) -> None:
    files = sorted(dataset_dir.rglob("*.npz"))
    if not files:
        logger.warning(f"No .npz files found under {dataset_dir}")
        return

    changed = 0
    for file in files:
        instance = CachedInstance.load(file)
        new_map_file = to_portable_path(instance.map_file)
        new_scen_file = to_portable_path(instance.scen_file)
        if new_map_file == instance.map_file and new_scen_file == instance.scen_file:
            continue

        changed += 1
        logger.info(
            f"{file}: map_file {instance.map_file!r} -> {new_map_file!r}, "
            f"scen_file {instance.scen_file!r} -> {new_scen_file!r}"
        )
        if not dry_run:
            instance.map_file = new_map_file
            instance.scen_file = new_scen_file
            instance.save(file)

    logger.info(f"{'Would change' if dry_run else 'Changed'} {changed}/{len(files)} files")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite cached dataset paths (map_file/scen_file) to be repo-relative."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory to scan recursively for .npz instance files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without modifying any files.",
    )
    args = parser.parse_args()
    migrate(args.dataset_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
