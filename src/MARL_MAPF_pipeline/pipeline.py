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


def run_pipeline(args: argparse.Namespace) -> None:
    if args.model_file is not None:
        model: DistanceTableCNN = load_model(args.model_file)
    elif not args.use_lacam_only:
        # TODO: Train this model so that it outputs distance tables with max distance in each cell
        model = DistanceTableCNN(lr=args.lr)

    # define problem instance
    grid = get_grid(args.map_file)
    starts, goals = get_scenario(args.scen_file, args.num_agents)

    if args.use_lacam_only:
        # Use LaCAM only without training or model
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
