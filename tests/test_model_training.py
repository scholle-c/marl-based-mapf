import pytest

from marl_path.model.evaluation import get_soc
import torch
import numpy as np
from marl_path.model.utils import build_random_input_tensor
from marl_path.model.definition import DistanceTableCNN
from marl_path.model.training import pretrain_on_default_value


SKIP_PRETRAIN_TEST = True


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
