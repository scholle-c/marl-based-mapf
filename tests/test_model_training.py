import pytest

from pathfinding_model.model_training import get_soc


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
