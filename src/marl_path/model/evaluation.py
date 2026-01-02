"""
Evaluation functions for MARL training runs and MAPF solutions.
"""

from __future__ import annotations
from typing import Any


def get_soc(solution: Any) -> int:
    """
    Compute the sum of costs (SOC) of a MAPF solution. You look for each agent
    for the last time step where the agent moves. Waiting is added to the cost,
    if the agent hasn't reached its goal yet. If the agent reached its goal, waiting
    at the goal does not add to the cost.

    All those individual costs are summed up to get the SOC.

    Args:
        solution: MAPF solution as a list of configurations.

    Returns:
        Sum of costs of the solution.
    """
    soc = 0
    num_agents = len(solution[0])
    for agent_idx in range(num_agents):
        # Find the last time step where the agent moves
        costs = 0
        goal = solution[-1][agent_idx]

        for t in range(1, len(solution)):
            if not (solution[t][agent_idx] == solution[t - 1][agent_idx] == goal):
                costs += 1
        soc += costs
    return soc
