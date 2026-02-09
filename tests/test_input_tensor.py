import pytest

from marl_path.model.evaluation import get_soc
import torch
import numpy as np
from marl_path.model.utils import build_random_input_tensor, build_input_tensor, _add_coords


def test_add_coords():
    """ Test the _add_coords function for correctness. """
    batch_size = 2
    channels = 3
    height = 4
    width = 5

    input_tensor = torch.zeros((batch_size, channels, height, width))
    output_tensor = _add_coords(input_tensor)

    assert output_tensor.shape == (batch_size, channels + 2, height, width)

    # Check coordinate values
    x_coords = output_tensor[:, -2, :, :].numpy()
    y_coords = output_tensor[:, -1, :, :].numpy()

    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                expected_x = (w / (width - 1)) * 2 - 1
                expected_y = (h / (height - 1)) * 2 - 1
                assert np.isclose(x_coords[b, h, w], expected_x), f"Batch {b}, Height {h}, Width {w}: Expected x {expected_x}, got {x_coords[b, h, w]}"
                assert np.isclose(y_coords[b, h, w], expected_y), f"Batch {b}, Height {h}, Width {w}: Expected y {expected_y}, got {y_coords[b, h, w]}"
