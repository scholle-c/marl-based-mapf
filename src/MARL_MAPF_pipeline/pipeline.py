import argparse
from pathfinding_model import DistanceTableCNN, load_model, train_on_solution, get_soc, pretrain_model_on_default_value
from pycam.mapf_utils import get_grid
from pycam import (
    LaCAM,
    get_scenario,
    validate_mapf_solution,
)
import torch
import os
import json
import numpy as np
from .constants import TRAIN_MODE_MODEL, TRAIN_MODE_LACAM_ONLY, TRAIN_MODE_BEST


def run_pipeline(args: argparse.Namespace) -> None:
    grid = get_grid(args.map_file)
    starts, goals = get_scenario(args.scen_file, args.num_agents)

    if args.model_file is not None:
        model: DistanceTableCNN = load_model(args.model_file)
    elif args.training_mode != TRAIN_MODE_LACAM_ONLY:
        model = DistanceTableCNN()
        #pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #pretrain_model_on_default_value(model, grid, pretrain_optimizer, num_epochs=1000)

    if args.training_mode == TRAIN_MODE_LACAM_ONLY:
        _run_lacam_only(args)
        return

    socs = []
    soc_model = []
    soc_no_model = []
    losses = []
    dist_table_differences = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    solution_found = False

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
        if len(solution) != 0:
            solution_found = True
            validate_mapf_solution(grid, starts, goals, solution)
        
        dist_tables_model = planner.dist_tables

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
            soc_without_model = get_soc(solution_no_model)

            dist_table_difference = np.mean([np.abs(dt_model.table - dt_no_model.table).mean().item() for dt_model, dt_no_model in zip(dist_tables_model, planner.dist_tables)])

            if not solution_found:
                solution = solution_no_model
                soc_with_model = np.nan
            else:
                soc_with_model = get_soc(solution)
                if soc_without_model < soc_with_model:
                    solution = solution_no_model
            
            dist_table_differences.append(dist_table_difference)
            soc_model.append(soc_with_model)
            soc_no_model.append(soc_without_model)

        soc = get_soc(solution)
        socs.append(soc)

        # train model
        mean_loss = train_on_solution(model, optimizer, solution, starts, goals, grid, device=args.device)
        losses.append(mean_loss)
        print(f"  SOC: {soc}, Mean Loss: {mean_loss:.4f}")
        solution_found = False
    
    print("Training completed.")
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "trained_model.pt")
        torch.save(model.state_dict(), model_path)
        json_path = os.path.join(args.output_dir, "training_stats.json")
        data = {"socs": socs, "losses": losses}
        if args.training_mode == TRAIN_MODE_BEST:
            data["soc_model"] = soc_model
            data["soc_no_model"] = soc_no_model
            data["dist_table_differences"] = dist_table_differences
        with open(json_path, "w") as f:
            json.dump(data, f)

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