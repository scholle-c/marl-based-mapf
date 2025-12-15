"""
Contains methods for reinforcement learning using LaCAM solutions, metrics computation, and model training.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from .distance_table_cnn import DistanceTableCNN
from .model_utils import build_input_tensor

if TYPE_CHECKING:
    from pycam.mapf_utils import Configs, Config, Coord, Grid


def train_on_solution(model: DistanceTableCNN, optimizer: torch.optim.Optimizer, solution: Configs, starts: Config, goals: Config, map: Grid, device: str = "auto") -> None:
    """
    RL fine-tuning based on a LaCAM solution.

    Args:
        model: Distance table CNN to update.
        optimizer: Optimizer for updating the model parameters.
        solution: Output of the planner as a list of configurations. Each element 
                    represents a time step, the first configuration is the start configuration, and the last configuration is the goal configuration.
        starts: Start configuration for each agent.
        goals: Goal configuration for each agent.
        map: Grid map of the environment.
    """
    model.train()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    optimizer.zero_grad()

    num_agents = len(starts)
    num_time_steps = len(solution)

    values = []
    targets = []

    # TODO: For each step of an agent, look at all neighbors and compute the target value of that neighbor as min(neighbor_values) + 1. Check if the neighbor is inside the map! You could also do the following: 0. Mask the distance table with the map layout 1. Copy the distance table 2. Replace values inside the path with the path distances from the agent 3. calculate the target values for the neighbors 4. Compute loss in a vectorized manner
    for agent_idx in range(num_agents):
        start = starts[agent_idx]
        goal = goals[agent_idx]
        input_tensor = build_input_tensor(map, goal, start).to(device)
        dist_table = model(input_tensor).squeeze(0).squeeze(0)

        values.append(dist_table[goal])
        targets.append(torch.tensor(0.0, device=device))

        for t in range(num_time_steps - 1, 0, -1):
            current_config = solution[t]
            prev_config = solution[t - 1]

            pos_t = current_config[agent_idx]
            pos_t_prev = prev_config[agent_idx]
            
            if pos_t == goal:
                v_t = torch.tensor(0.0, device=device)
            else:
                v_t = dist_table[pos_t]
            v_t_prev = dist_table[pos_t_prev]

            target = v_t.detach() + 1.0

            values.append(v_t_prev)
            targets.append(target)

    values_tensor = torch.stack(values)
    targets_tensor = torch.stack(targets)

    mean_loss = torch.nn.functional.mse_loss(values_tensor, targets_tensor)
    mean_loss.backward()
    optimizer.step()
    return mean_loss.item()

def get_soc(solution: Configs) -> int:
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
