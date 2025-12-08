"""
Stub for future reinforcement learning step using LaCAM solutions.
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
    total_loss = 0.0

    num_samples = 0
    num_agents = len(starts)
    num_time_steps = len(solution)

    for agent_idx in range(num_agents):
        start = starts[agent_idx]
        goal = goals[agent_idx]
        input_tensor = build_input_tensor(map, goal, start).to(device)
        dist_table = model(input_tensor).squeeze(0).squeeze(0)

        for t in range(1, num_time_steps):
            current_config = solution[t - 1]
            next_config = solution[t]

            pos_t = current_config[agent_idx]
            pos_t1 = next_config[agent_idx]

            v_t = dist_table[pos_t]
            v_t1 = dist_table[pos_t1]

            # goal handling: if s_t is already the goal, target=0
            if pos_t == goal:
                target = torch.tensor(0.0, device=device)
            else:
                target = 1.0 + v_t1.detach()
            
            num_samples += 1
            loss = (v_t - target) ** 2
            total_loss += loss

    mean_loss = total_loss / num_samples
    mean_loss.backward()
    optimizer.step()
    return mean_loss.item()

def get_soc(solution: Configs) -> int:
    """
    Compute the sum of costs (SOC) of a MAPF solution.

    Args:
        solution: MAPF solution as a list of configurations.

    Returns:
        Sum of costs of the solution.
    """
    soc = 0
    num_agents = len(solution[0])
    for agent_idx in range(num_agents):
        # Find the last time step where the agent moves
        last_step = 0
        for t in range(len(solution)):
            if solution[t][agent_idx] != solution[t - 1][agent_idx]:
                last_step = t
        soc += last_step
    return soc
