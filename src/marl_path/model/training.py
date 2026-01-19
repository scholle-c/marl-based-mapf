"""
Everything related to model training. Contains functions like updating the model weights.
"""

from __future__ import annotations
from typing import Any, Tuple, List, Dict
import torch
import numpy as np

from .utils import build_input_tensor, build_random_input_tensor
from marl_path.shared import get_neighbors, Coord


def train_on_lacam_solution(
    model: Any,
    optimizer: Any,
    solution: Any,
    starts: Any,
    goals: Any,
    map: Any,
    device: torch.device | None = None,
    use_neighbors: bool = False,
    goal_weight: float = 1.0,
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
        use_neighbors: Whether to include neighboring cells in the loss computation.
        goal_weight: Weight for samples at the goal (target == 0).
    """

    model.train()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer.zero_grad()

    num_agents = len(starts)

    values: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for agent_idx in range(num_agents):
        path = [solution[t][agent_idx] for t in range(len(solution))]
        agent_values, agent_targets = _get_agent_values_targets(
            starts[agent_idx],
            goals[agent_idx],
            path,
            map,
            model,
            device,
            use_neighbors=use_neighbors,
        )

        values.extend(agent_values)
        targets.extend(agent_targets)

    values_tensor = torch.stack(values)
    targets_tensor = torch.stack(targets)
    weights_tensor = torch.ones_like(targets_tensor)
    weights_tensor = torch.where(
        targets_tensor == 0,
        torch.tensor(
            goal_weight, device=targets_tensor.device, dtype=targets_tensor.dtype
        ),
        weights_tensor,
    )

    loss = torch.nn.functional.mse_loss(values_tensor, targets_tensor, reduction="none")
    # Emphasize the goal position (target == 0) via weighting.
    mean_loss = (loss * weights_tensor).mean()
    mean_loss.backward()
    optimizer.step()
    return mean_loss.item()


def _get_agent_values_targets(
    start: Coord,
    goal: Coord,
    path: Any,
    map: Any,
    model: Any,
    device: Any,
    use_neighbors: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Takes in the agents path from start to goal and the distance table, that was used
    as a heuristic to navigate the map. Creates two tensors out of it:

    - Values: Contains the predicted distances
    - Targets: Contains the actual distances

    This can then be used to do reinforcement learning to get better distance table
    predictions.

    If wanted, the user can also include the neighbors of the path into the tensors.
    As a target value, the minimum distance of all neighbors plus one is used. This
    adds some sort of exploration to other areas of the map.

    Args:
        start (Any): The start coordinate of the agent
        goal (Any): The goal coordinate of the agent
        path (Any): The path the agent took from start to goal
        map (Any): The map as a 2D numpy array
        model (Any): The distance table model used to predict the distances
        device (Any): The device to run the model on
        use_neighbors (bool, optional): Whether to include neighbors in the tensors. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The predicted distances (Values) and the actual distances (Targets).
    """
    dist_table = (
        model(build_input_tensor(map, goal, start).to(device)).squeeze(0).squeeze(0)
    )
    return _get_agent_values_targets_helper(
        path, map, dist_table, device, use_neighbors
    )


def _get_agent_values_targets_helper(
    path: Any,
    map: Any,
    dist_table: torch.Tensor,
    device: Any,
    use_neighbors: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function for _get_agent_values_targets, see there for documentation.
    """
    values: torch.Tensor = _get_via_coordinates(dist_table, path)
    targets: torch.Tensor = torch.arange(
        len(path) - 1, -1, -1, device=device, dtype=torch.float32, requires_grad=False
    )

    if use_neighbors:
        neighbors: Dict = _get_neighbors_of_path(map, path)
        if len(neighbors) == 0:
            return values, targets

        neigh_coords: List[Coord] = list(neighbors.keys())
        values_neigh: torch.Tensor = _get_via_coordinates(dist_table, neigh_coords)
        target_neigh: List[torch.Tensor] = []
        dist_table_arr: np.ndarray = dist_table.detach().cpu().numpy()

        for neigh in neigh_coords:
            target_neigh.append(
                _get_neighbor_target(neighbors[neigh], path, dist_table_arr, device)
            )

        values = torch.cat((values, values_neigh), dim=0)
        targets = torch.cat((targets, torch.stack(target_neigh)), dim=0)

    return values, targets


def _get_via_coordinates(arr: Any, coords: List[Coord]) -> Any:
    """
    Accesses the elements of an 2D numpy array via a list of 2D coordinates in the
    shape [(column, row), ...].

    Args:
        arr (np.ndarray): The 2D numpy array to access.
        coords (List[Coord]): The list of 2D coordinates to access.

    Returns:
        np.ndarray: The elements of the array at the specified coordinates.
    """
    if len(coords) == 0:
        if isinstance(arr, torch.Tensor):
            return torch.tensor([], dtype=arr.dtype, device=arr.device)
        return np.array([], dtype=arr.dtype)
    idx = tuple(np.array(coords).T)
    return arr[idx]


def _get_neighbors_of_path(
    map: Any, path: Any
) -> Dict[Coord, Tuple[List[Coord], List[Coord]]]:
    """
    Gives you a dictionary, containing the coordinates of adjacent tiles to the given
    path as keys. As values, it contains a tuple of two lists: the first list contains the coordinates of neighbors that are part of the path,
    the second list contains the coordinates of neighbors that are not part of the path.


    Args:
        map (Any): The map as a 2D numpy array
        path (Any): The path, for which the neighbor coordinates should be found

    Returns:
        Dict[Tuple[int, int], Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]: A dict containing the neighbor coords as keys and as Values a Tuple of two lists containing on-path and off path neighbors.
    """
    neighbors: set = set()

    # First loop: Get all neighbors next to the path
    for pos in path:
        neighbor_positions = get_neighbors(map, pos)
        neighbors.update(
            [
                tuple(neighbor)
                for neighbor in neighbor_positions
                if tuple(neighbor) not in path and map[neighbor] == 1
            ]
        )

    neighbor_dict: dict = {}

    # Second loop: For each neighbor, get its neighbors and assign path and non-path neighbors
    for neighbor in neighbors:
        neighbor_positions = get_neighbors(map, neighbor)
        path_neighbors: list = []
        non_path_neighbors: list = []
        for pos in neighbor_positions:
            if tuple(pos) in path:
                path_neighbors.append(tuple(pos))
            else:
                non_path_neighbors.append(tuple(pos))
        neighbor_dict[neighbor] = (path_neighbors, non_path_neighbors)

    return neighbor_dict


def _get_neighbor_target(
    neighbor_coords: Tuple[List[Coord], List[Coord]],
    path: Any,
    dist_table: np.ndarray,
    device: Any,
) -> torch.Tensor:
    """
    Determines the target value of a coordinate, by looking at all neighbor-coordinates of it,
    and taking the minimum value of them plus one.

    Detailed behavior:
    - For neighbors that are not part of the path, the value is directly taken from the distance table.
    - For neighbors that are part of the path, the value is determined by looking at all positions in the path
      where this coordinate is visited, and taking the index of that position in the path as the value.
    - Then the minimum of all these values is taken.

    Args:
        neighbor_coords (Tuple[List[Coord], List[Coord]]): Neighbor coordinates as a tuple of two lists: First list contains on-path neighbors, second list contains non-path neighbors.
        path (Any): The path, which is used to determine the target values for on-path neighbors.
        dist_table (np.ndarray): The distance table.
        device (Any): The device on which the resulting tensor should be.

    Returns:
        torch.Tensor: The target value for the coordinate.
    """
    # Get values of non-path neighbors
    targets: np.ndarray = _get_via_coordinates(dist_table, neighbor_coords[1])
    targets_on_path: list = []

    # Special case on path neighbors: Here a coordinate might be visited multiple times in the path
    for on_path_pos in neighbor_coords[0]:
        targets_on_path.append(
            [
                tgt
                for tgt, pos in enumerate(reversed(path))
                if tuple(pos) == on_path_pos
            ][0]
        )

    targets = np.append(targets, targets_on_path)
    target = min(targets) + 1.0
    return torch.tensor(target, device=device, dtype=torch.float32, requires_grad=False)


def pretrain_on_default_value(
    model: Any,
    grid: Any,
    optimizer: Any,
    default_value: int | None = None,
    num_epochs: int = 10,
    device: torch.device | None = None,
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

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fill_value: int = grid.size if default_value is None else default_value
    target_tensor: torch.Tensor = torch.full(
        size=grid.shape, fill_value=fill_value, dtype=torch.float32, device=device
    )

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        random_input: torch.Tensor = build_random_input_tensor(grid, device=device)
        value_tensor = model(random_input).squeeze(0).squeeze(0)
        mean_loss = torch.nn.functional.mse_loss(value_tensor, target_tensor)
        mean_loss.backward()
        optimizer.step()
