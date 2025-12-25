"""
Contains methods for reinforcement learning using LaCAM solutions, metrics computation, and model training.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from .distance_table_cnn import DistanceTableCNN
from .model_utils import build_input_tensor, build_random_input_tensor

import numpy as np
from math import inf

from mapf_utils import Configs, Config, Coord, Grid, get_neighbors


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

    visited_neighbors: set = set()
    visited_path: set = set()
    values = []
    targets = []

    for agent_idx in range(num_agents):
        start = starts[agent_idx]
        goal = goals[agent_idx]
        input_tensor = build_input_tensor(map, goal, start).to(device)
        dist_table = model(input_tensor).squeeze(0).squeeze(0)
        dist_table_paths = dist_table.detach().clone()
        dist_table_paths[goal] = 0

        visited_path.add(goal)

        values.append(dist_table[goal])
        targets.append(torch.tensor(0.0, device=device))

        i = 1
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

            # Is later used for neighbor distance calculation
            # TODO: Bisher wird der Fall nicht abgefangen, wenn ein agent zweimal dasselbe Feld passiert (Hier könnte dann z.b. der kleinere Wert genommen werden)
            dist_table_paths[pos_t_prev] = i
            i += 1
            neighbors = get_neighbors(map, pos_t_prev)
            visited_neighbors.update(neighbors)
            visited_path.add(pos_t_prev)

    # Now compute neighbor values and add them to the loss
    visited_neighbors = visited_neighbors - visited_path

    for neighbor in visited_neighbors:
        min_dist = inf
        for pos in get_neighbors(map, neighbor):
            if dist_table_paths[pos] < min_dist:
                min_dist = dist_table_paths[pos]
        if min_dist == inf:
            continue
        values.append(dist_table[neighbor])
        targets.append(torch.tensor(min_dist.detach().clone() + 1.0, device=device))

    values_tensor = torch.stack(values)
    targets_tensor = torch.stack(targets)

    mean_loss = torch.nn.functional.mse_loss(values_tensor, targets_tensor)
    mean_loss.backward()
    optimizer.step()
    return mean_loss.item()

def pretrain_model_on_default_value(model: DistanceTableCNN, grid: Grid, optimizer: torch.optim.Optimizer, default_value: int = None, num_epochs: int = 10) -> None:
    """
    Trains the distance-table model, to predict a default value
    for random input. Can be used as a way of initialization.

    Args:
        model (DistanceTableCNN): The model that should be trained
        grid (Grid): The map that the model should be trained on
        default_value (float, optional): The default value that should be predicted. Default is the map size.
        num_epochs (int, optional): How many epochs should be used for training. Defaults to 10.
    """
    # TODO: Hier weitermachen, sprich: 1. Braucht es ein äqquivalent zu train()? --> Nein 2. Schreib test zu dem Code 3. Debugg den code, der wird so noch nicht laufen 4. Gibts Modelling tools in python fürs refactoring?
    device = next(model.parameters()).device
    
    if default_value is None:
        default_value = grid.size

    target_tensor: torch.tensor = torch.full(grid.shape, default_value, dtype=torch.float32, device=device)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        random_input: torch.Tensor = build_random_input_tensor(grid)
        value_tensor = model(random_input).squeeze(0).squeeze(0)
        mean_loss = torch.nn.functional.mse_loss(value_tensor, target_tensor)
        mean_loss.backward()
        optimizer.step()
    
    
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
