"""
Everything related to model training. Contains functions like updating the model weights.
"""

from __future__ import annotations
from typing import Any
import torch

from .utils import build_input_tensor, build_random_input_tensor
from marl_path.shared import get_neighbors


def train_on_lacam_solution(
    model: Any,
    optimizer: Any,
    solution: Any,
    starts: Any,
    goals: Any,
    map: Any,
    device_str: str | None = None,
) -> float:
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
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    optimizer.zero_grad()

    num_agents = len(starts)
    num_time_steps = len(solution)

    values = []
    targets = []

    for agent_idx in range(num_agents):
        start = starts[agent_idx]
        goal = goals[agent_idx]
        input_tensor = build_input_tensor(map, goal, start).to(device)
        dist_table = model(input_tensor).squeeze(0).squeeze(0)
        dist_table_paths = dist_table.detach().clone()
        dist_table_paths[goal] = 0

        visited_neighbors: set = set()
        visited_path: set = set()
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
            min_dist: torch.Tensor | None = None
            for pos in get_neighbors(map, neighbor):
                candidate = dist_table_paths[pos]
                if min_dist is None or candidate < min_dist:
                    min_dist = candidate
            if min_dist is None:
                continue
            values.append(dist_table[neighbor])
            targets.append(torch.tensor(min_dist.detach().clone() + 1.0, device=device))

    values_tensor = torch.stack(values)
    targets_tensor = torch.stack(targets)

    mean_loss = torch.nn.functional.mse_loss(values_tensor, targets_tensor)
    mean_loss.backward()
    optimizer.step()
    return mean_loss.item()


def pretrain_on_default_value(
    model: Any,
    grid: Any,
    optimizer: Any,
    default_value: int | None = None,
    num_epochs: int = 10,
) -> None:
    """
    Trains the distance-table model, to predict a default value
    for random input. Can be used as a way of initialization.

    Args:
        model (DistanceTableCNN): The model that should be trained
        grid (Grid): The map that the model should be trained on
        default_value (float, optional): The default value that should be predicted. Default is the map size.
        num_epochs (int, optional): How many epochs should be used for training. Defaults to 10.
    """

    device = next(model.parameters()).device
    fill_value: int = grid.size if default_value is None else default_value
    target_tensor: torch.Tensor = torch.full(
        size=grid.shape, fill_value=fill_value, dtype=torch.float32, device=device
    )

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        random_input: torch.Tensor = build_random_input_tensor(grid)
        value_tensor = model(random_input).squeeze(0).squeeze(0)
        mean_loss = torch.nn.functional.mse_loss(value_tensor, target_tensor)
        mean_loss.backward()
        optimizer.step()
