import numpy as np
import torch

from pathfinding_model.distance_table_cnn import DistanceTableCNN


def test_set_default_output_value_outputs_constant_map():
    model = DistanceTableCNN()
    side_length = 4
    constant_value = float(side_length * side_length)  # map.size
    rng = np.random.default_rng(0)

    map_channel = torch.ones((side_length, side_length), dtype=torch.float32)
    goal_channel = torch.zeros_like(map_channel)
    start_channel = torch.zeros_like(map_channel)

    goal = tuple(rng.integers(0, side_length, size=2))
    start = tuple(rng.integers(0, side_length, size=2))
    goal_channel[goal] = 1.0
    start_channel[start] = 1.0

    input_tensor = torch.stack([map_channel, goal_channel, start_channel], dim=0).unsqueeze(0)

    model.set_default_output_value(constant_value)
    with torch.no_grad():
        output = model(input_tensor).squeeze()

    expected = torch.full((side_length, side_length), constant_value)
    assert output.shape == expected.shape
    assert torch.allclose(output, expected)
