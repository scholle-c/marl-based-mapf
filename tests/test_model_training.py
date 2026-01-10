import pytest

from marl_path.model.evaluation import get_soc
import torch
import numpy as np
from marl_path.model.utils import build_random_input_tensor
from marl_path.model.definition import DistanceTableCNN
from marl_path.model.training import (
    pretrain_on_default_value,
    _get_via_coordinates,
    _get_neighbors_of_path,
    _get_agent_values_targets_helper,
)


SKIP_PRETRAIN_TEST = True


def test_get_agent_values_targets():
    """
    Test the _get_agent_values_targets function with a sample path and distance table.
    """

    # Sample path: (y, x) coordinates
    path = [(2, 0), (1, 0), (1, 1), (1, 0), (0, 0)]

    # Sample map
    map_array = np.array([[1, 1], [1, 1], [1, 1]], dtype=bool)

    # Sample distance table as a torch tensor
    dist_table = torch.tensor([[1, 6], [2, 0], [1, 6]], dtype=torch.float32)

    # Expected values and targets
    expected_values = torch.tensor([1.0, 2.0, 0.0, 2.0, 1.0], dtype=torch.float32)
    expected_targets = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0], dtype=torch.float32)

    # Call the function
    values, targets = _get_agent_values_targets_helper(
        path, map_array, dist_table, device="cpu", use_neighbors=False
    )

    # Assertions
    assert torch.equal(values, expected_values), (
        f"Expected values {expected_values}, got {values}"
    )
    assert torch.equal(targets, expected_targets), (
        f"Expected targets {expected_targets}, got {targets}"
    )

    # Do the same test but with neighbors
    expected_values_with_neighbors = torch.tensor(
        [1.0, 2.0, 0.0, 2.0, 1.0, 6.0, 6.0], dtype=torch.float32
    )
    expected_targets_with_neighbors = torch.tensor(
        [4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 3.0], dtype=torch.float32
    )

    values_with_neighbors, targets_with_neighbors = _get_agent_values_targets_helper(
        path, map_array, dist_table, device="cpu", use_neighbors=True
    )

    assert torch.equal(values_with_neighbors, expected_values_with_neighbors), (
        f"Expected values with neighbors {expected_values_with_neighbors}, got {values_with_neighbors}"
    )
    assert torch.equal(targets_with_neighbors, expected_targets_with_neighbors), (
        f"Expected targets with neighbors {expected_targets_with_neighbors}, got {targets_with_neighbors}"
    )


def test_get_via_coordinates():
    """
    Test the _get_via_coordinates function with a sample 2D array and coordinates.
    """

    arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    coords = [(0, 1), (1, 2), (2, 0)]  # Should access elements 20, 60, 70

    expected_values = np.array([20, 60, 70])
    values = _get_via_coordinates(arr, coords)

    assert np.array_equal(values, expected_values), (
        f"Expected {expected_values}, got {values}"
    )


def test_get_neighbors_of_path():
    """
    Test the _get_neighbors_of_path function with a sample map and path. As a reminder,
    the path is given via (y, x) - coordinates.

    So the path in this example would be (numbers indicate the order of the path):
    -----
    |1oo|
    |234|
    |xxo|
    -----
    """
    map_array = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=bool)
    path = [(0, 0), (1, 0), (1, 1), (1, 2)]

    expected_neighbors = {
        (0, 1): ([(0, 0), (1, 1)], [(0, 2)]),
        (0, 2): ([(1, 2)], [(0, 1)]),
        (2, 2): ([(1, 2)], []),
    }

    neighbors = _get_neighbors_of_path(map_array, path)

    assert neighbors == expected_neighbors, (
        f"Expected {expected_neighbors}, got {neighbors}"
    )


def test_get_neighbors_of_path_no_neighbors():
    """
    Test the _get_neighbors_of_path function with a sample map and path where no neighbors are available.
    """
    map_array = np.array([[1, 1], [1, 1]], dtype=bool)
    path = [(0, 0), (0, 1), (1, 0), (1, 1)]

    expected_neighbors = {}

    neighbors = _get_neighbors_of_path(map_array, path)

    assert neighbors == expected_neighbors, (
        f"Expected {expected_neighbors}, got {neighbors}"
    )


def test_get_soc():
    """
    Setup:
        Agents: 2
        Goals: Agent 0 -> (0,2), Agent 1 -> (1,3)
        Costs:
            Agent 0: Moves at t=1, t=2; waits in goal at t=3, t=4 -> Cost = 2
            Agent 1: Waits at t=1; t=2; moves at t=3, t=4 -> Cost = 4
            SOC = 2 + 4 = 6
    """
    solution = [
        [(0, 0), (1, 1)],  # t = 0
        [(0, 1), (1, 1)],  # t = 1
        [(0, 2), (1, 1)],  # t = 2
        [(0, 2), (1, 2)],  # t = 3
        [(0, 2), (1, 3)],  # t = 4
    ]

    expected_soc = 6  # Expected sum of costs for this solution

    soc = get_soc(solution)
    assert soc == expected_soc


def test_pretrain_model_on_default_value():
    if SKIP_PRETRAIN_TEST:
        pytest.skip("Skipping pretrain model test to save time.")

    # arrange
    model = DistanceTableCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    grid_array = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )
    default_value = 9

    # act
    pretrain_on_default_value(
        model, grid_array, optimizer, default_value=default_value, num_epochs=1000
    )

    # assert
    model.eval()
    with torch.no_grad():
        random_input = build_random_input_tensor(grid_array)
        output = model(random_input).squeeze().cpu().numpy()
        mean_output_value = np.mean(output)
        assert np.isclose(mean_output_value, default_value, atol=1.0), (
            f"Mean output value {mean_output_value} not close to default value {default_value}"
        )
