import argparse
from pathfinding_model import DistanceTableCNN, load_model, train_on_solution, get_soc
from pycam.mapf_utils import get_grid
from pycam import (
    LaCAM,
    get_scenario,
    validate_mapf_solution,
)
import torch
import os
import json
from .constants import TRAIN_MODE_MODEL, TRAIN_MODE_LACAM_ONLY, TRAIN_MODE_BEST


def run_pipeline(args: argparse.Namespace) -> None:
    grid = get_grid(args.map_file)
    starts, goals = get_scenario(args.scen_file, args.num_agents)

    if args.model_file is not None:
        model: DistanceTableCNN = load_model(args.model_file)
    elif args.training_mode != TRAIN_MODE_LACAM_ONLY:
        model = DistanceTableCNN(lr=args.lr)
        model.set_default_output_value(grid.size)

    if args.training_mode == TRAIN_MODE_LACAM_ONLY:
        _run_lacam_only(args)
        return

    socs = []
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Start training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # solve MAPF
        planner = LaCAM()
        
        solution = planner.solve(
            grid=grid,
            starts=starts,
            goals=goals,
            model=model,
            seed=args.seed + epoch,
            time_limit_ms=args.time_limit_ms,
            flg_star=args.flg_star,
            verbose=args.verbose,
        )
        validate_mapf_solution(grid, starts, goals, solution)

        if args.training_mode == TRAIN_MODE_BEST:
            # Solve again without model
            solution_no_model = planner.solve(
                grid=grid,
                starts=starts,
                goals=goals,
                model=None,
                seed=args.seed + epoch,
                time_limit_ms=args.time_limit_ms,
                flg_star=args.flg_star,
                verbose=args.verbose,
            )
            validate_mapf_solution(grid, starts, goals, solution_no_model)

            soc_with_model = get_soc(solution)
            soc_without_model = get_soc(solution_no_model)

            if soc_without_model < soc_with_model:
                solution = solution_no_model

        soc = get_soc(solution)
        socs.append(soc)

        # train model
        mean_loss = train_on_solution(model, optimizer, solution, starts, goals, grid, device=args.device)
        losses.append(mean_loss)
        print(f"  SOC: {soc}, Mean Loss: {mean_loss:.4f}")
    
    print("Training completed.")
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "trained_model.pt")
        torch.save(model.state_dict(), model_path)
        json_path = os.path.join(args.output_dir, "training_stats.json")
        with open(json_path, "w") as f:
            json.dump({"socs": socs, "losses": losses}, f)

def _run_lacam_only(args: argparse.Namespace) -> None:
    """
    Run LaCAM once without any rl training or using a distance table CNN model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # define problem instance
    grid = get_grid(args.map_file)
    starts, goals = get_scenario(args.scen_file, args.num_agents)

    planner = LaCAM()
    solution = planner.solve(
        grid=grid,
        starts=starts,
        goals=goals,
        model=None,
        seed=args.seed,
        time_limit_ms=args.time_limit_ms,
        flg_star=args.flg_star,
        verbose=args.verbose,
    )
    validate_mapf_solution(grid, starts, goals, solution)
    soc = get_soc(solution)
    print(f"LaCAM only SOC: {soc}")