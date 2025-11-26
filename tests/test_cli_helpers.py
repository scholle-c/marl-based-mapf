import numpy as np
import pytest

from pathfinding_model_training.cli import _save_generated_data


def test_save_generated_data_uses_map_stems(tmp_path):
    inputs = np.zeros((1, 3, 1, 1), dtype=np.int8)
    labels = np.zeros((1, 1, 1), dtype=float)
    datasets = [(inputs, labels), (inputs, labels)]
    map_paths = ["./assets/tunnel.map", "./assets/tunnel.map"]

    saved = _save_generated_data(datasets, tmp_path, map_paths)
    expected = {tmp_path / "training_data_tunnel.npz", tmp_path / "training_data_tunnel_1.npz"}

    assert set(saved) == expected
    for path in saved:
        assert path.exists()


def test_save_generated_data_validates_length(tmp_path):
    inputs = np.zeros((1, 3, 1, 1), dtype=np.int8)
    labels = np.zeros((1, 1, 1), dtype=float)
    datasets = [(inputs, labels)]
    map_paths = ["./assets/tunnel.map", "./assets/corners.map"]

    with pytest.raises(ValueError):
        _save_generated_data(datasets, tmp_path, map_paths)
