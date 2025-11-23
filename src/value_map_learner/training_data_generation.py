"""Utilities to generate training data from MAPF grid files. The training data 
can be used to train a neural network to learn a value map"""

from collections import deque
from pathlib import Path
from typing import List

import argparse

import numpy as np

from .mapf_utils import get_grid, get_neighbors


def create_training_data(map_file_paths: List[str]) -> List[tuple[np.ndarray, np.ndarray]]:
    """
    Load maps and generate training data for each map.

    Parameters
    ----------
    map_file_paths:
        Sequence of file paths pointing to map descriptions that can be fed to
        `mapf_utils.get_grid`.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of `(inputs, labels)` pairs, one per map.
    """

    grids = (_load_grid(path) for path in map_file_paths)
    return [create_training_data_for_map(grid) for grid in grids]


def _load_grid(path: str):
    """Thin wrapper that keeps `create_training_data` easy to test/mock."""
    return get_grid(path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training data from MAPF grid files."
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        required=True,
        help="Paths to map files consumed by get_grid().",
    )
    return parser.parse_args()


def get_all_possible_map_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    """
    Return all accessible coordinates on the map (cells with value True).

    Parameters
    ----------
    grid:
        Two-dimensional boolean numpy array representing the map.

    Returns
    -------
    list[tuple[int, int]]
        List of coordinates (y, x) for all accessible positions.
    """

    if grid.ndim != 2:
        raise ValueError("Grid must be a 2D array.")

    return [tuple(coord) for coord in np.argwhere(grid)]


def create_distance_table(grid: np.ndarray, goal_coordinate: tuple[int, int]) -> np.ndarray:
    """
    Create a distance map from the goal coordinate using BFS.

    Parameters
    ----------
    grid:
        Two-dimensional boolean numpy array where True indicates accessible cells.
    goal_coordinate:
        The goal coordinate (y, x) from which to calculate distances.

    Returns
    -------
    np.ndarray
        A distance map with the same shape as `grid`, where each cell contains
        the shortest distance to the goal. Inaccessible or unreachable cells
        are assigned `grid.size`.
    """

    if not grid[goal_coordinate]:
        raise ValueError("Goal coordinate must be an accessible cell.")

    max_distance = grid.size

    distance_map = np.full(grid.shape, max_distance, dtype=float)
    distance_map[goal_coordinate] = 0

    queue = deque([goal_coordinate])
    while queue:
        current = queue.popleft()
        current_distance = distance_map[current]

        for neighbor in get_neighbors(grid, current):
            if distance_map[neighbor] == max_distance:
                distance_map[neighbor] = current_distance + 1
                queue.append(neighbor)

    return distance_map


def create_training_data_for_map(grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate input/label pairs for a single map by creating distance tables for all accessible positions.

    Parameters
    ----------
    grid:
        Two-dimensional boolean numpy array representing the map.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - Inputs with shape (num_positions, 3, height, width) where channels are map, goal, start.
        - Labels with shape (num_positions, height, width) containing distance maps.
    """
    # TODO: You could add features like "Create Samples...": for all start-goal combinations, for random start-goal pairs
    accessible_positions = get_all_possible_map_positions(grid)

    if not accessible_positions:
        empty_inputs = np.empty((0, 3) + grid.shape, dtype=np.int8)
        empty_labels = np.empty((0,) + grid.shape, dtype=float)
        return empty_inputs, empty_labels

    map_channel = grid.astype(np.int8, copy=False)
    rng = np.random.default_rng()

    input_samples: list[np.ndarray] = []
    label_samples: list[np.ndarray] = []

    for goal in accessible_positions:
        goal_channel = np.zeros_like(map_channel, dtype=np.int8)
        goal_channel[goal] = 1

        start_channel = np.zeros_like(map_channel, dtype=np.int8)
        start = accessible_positions[rng.integers(len(accessible_positions))]
        start_channel[start] = 1

        sample_input = np.stack((map_channel, goal_channel, start_channel), axis=0)
        input_samples.append(sample_input)

        distance_map = create_distance_table(grid, goal)
        label_samples.append(distance_map)

    inputs_array = np.stack(input_samples, axis=0)
    labels_array = np.stack(label_samples, axis=0)
    return inputs_array, labels_array


def save_training_data(inputs: np.ndarray, labels: np.ndarray, filepath: str | Path) -> None:
    """
    Save inputs and labels into a compressed NumPy `.npz` archive.

    Parameters
    ----------
    inputs:
        Input tensor, e.g. returned from `create_training_data_for_map`.
    labels:
        Label tensor (distance maps) corresponding to `inputs`.
    filepath:
        Destination file path where the archive should be stored.
    """

    path = Path(filepath)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(path, inputs=inputs, labels=labels)


def load_training_data(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load previously saved training data created by `save_training_data`.

    Parameters
    ----------
    filepath:
        Path to the `.npz` archive containing inputs and labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The stored `(inputs, labels)` pair.
    """

    with np.load(filepath) as data:
        inputs = data["inputs"]
        labels = data["labels"]
    return inputs, labels


if __name__ == "__main__":
    args = _parse_args()
    datasets = create_training_data(args.maps)
    total_samples = sum(inputs.shape[0] for inputs, _ in datasets)
    print(f"Created training data for {len(datasets)} maps with {total_samples} samples")
    
