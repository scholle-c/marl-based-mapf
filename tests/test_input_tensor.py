import pytest

from marl_path.model.evaluation import get_soc
import torch
import numpy as np
from marl_path.model.utils import build_random_input_tensor, build_input_tensor, _add_coords


def test_random_input_tensor():
    """ Test the build_random_input_tensor function for correctness. """
    grid = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
    tensor = build_random_input_tensor(grid, use_random_input_channels=False)

    assert tensor.shape == (1, 5, 3, 3)  # 3 channels + 2 for coordinates
    assert tensor.dtype == torch.float32    

    # Check that the map channel is correct
    map_channel = tensor[0, 0].numpy()
    assert np.array_equal(map_channel, grid.astype(np.float32))                                                                                                                                                                                                                                                                                                                    


def test_add_coords():
    """ Test the _add_coords function for correctness. """
    batch_size = 2
    channels = 3
    height = 4
    width = 5

    goal = (1, 1)  # Dummy goal for testing
    input_tensor = torch.zeros((batch_size, channels, height, width))
    output_tensor = _add_coords(input_tensor, goal)

    assert output_tensor.shape == (batch_size, channels + 2, height, width)

    # Check that the coordinate channels are correct
    x_coords = output_tensor[0, -2].numpy()
    y_coords = output_tensor[0, -1].numpy()
    for i in range(height):
        for j in range(width):
            assert x_coords[i, j] == j - goal[1]
            assert y_coords[i, j] == i - goal[0]
            