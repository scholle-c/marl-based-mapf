import pytest

from marl_path.model.evaluation import get_soc
import torch
import numpy as np
from marl_path.model.utils import build_random_input_tensor
from marl_path.model.definition import DistanceTableCNN
from marl_path.model.training import (
    pretrain_on_default_value,
    _get_via_coordinates,
    _get_targets_for_path,
    train_vdn_on_solution,
    _remove_waiting_from_path
)


SKIP_PRETRAIN_TEST = True


def test_get_path_target():
    """
    Test the _get_path_target function with a sample path and distance table.
    """

    # Sample path: (y, x) coordinates
    path = [(2, 0), (1, 0), (1, 0), (0, 0), (0, 0)]
    path2 = [(1,1), (1,2), (1,2), (1,3), (1,2), (2,2)]
    path_goal_overlapping = [(1, 0), (0, 0), (0, 1), (0, 1), (0, 0), (0, 0)]

    # Expected targets
    expected_targets = [2, 1, 0]
    expected_targets2 = [4, 1, 2, 0]
    expected_targets_goal_overlapping = [3, 0, 1]

    # Remove waiting from path and get targets for path
    path = _remove_waiting_from_path(path)
    path2 = _remove_waiting_from_path(path2)
    path_goal_overlapping = _remove_waiting_from_path(path_goal_overlapping)

    # Call the function
    targets = _get_targets_for_path(path)
    targets2 = _get_targets_for_path(path2)
    targets_goal_overlapping = _get_targets_for_path(path_goal_overlapping)
    # Assertions
    assert list(targets.values()) == expected_targets, (
        f"Expected targets {expected_targets}, got {targets}"
    )
    assert list(targets2.values()) == expected_targets2, (
        f"Expected targets {expected_targets2}, got {targets2}"
    )
    assert list(targets_goal_overlapping.values()) == expected_targets_goal_overlapping, (
        f"Expected targets {expected_targets_goal_overlapping}, got {targets_goal_overlapping}"
    )


def test_values_targets_match():
    """ Check whether the values are associated with the correct targets. """
    path = [(2, 1), (2, 0), (2, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)]

    dist_table = torch.tensor([[0, 1], 
                               [1, 2], 
                               [2, 3]], dtype=torch.float32)

    path = _remove_waiting_from_path(path)

    values: torch.Tensor = _get_via_coordinates(dist_table, path)
    targets_int = list(_get_targets_for_path(path).values())
    targets: torch.Tensor = torch.tensor(targets_int, dtype=torch.float32, device="cpu", requires_grad=False)

    for v, t in zip(values, targets):
        assert torch.isclose(v, t), f"Value {v.item()} does not match target {t.item()}"



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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # arrange
    model = DistanceTableCNN().to(device)
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
        model, grid_array, optimizer, default_value=default_value, num_epochs=1000, device=device
    )

    # assert
    model.eval()
    with torch.no_grad():
        random_input = build_random_input_tensor(grid_array, device=device)
        output = model(random_input).squeeze().cpu().numpy()
        mean_output_value = np.mean(output)
        assert np.isclose(mean_output_value, default_value, atol=1.0), (
            f"Mean output value {mean_output_value} not close to default value {default_value}"
        )
