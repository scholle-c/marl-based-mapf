import argparse
import random
from .model import (
    DistanceTableCNN,
    load_model,
    train_on_lacam_solution,
    get_soc,
    pretrain_on_default_value,
    TrainingStats,
)
from marl_path.shared.mapf_utils import get_grid, get_scenario, validate_mapf_solution
from .pycam import LaCAM
from typing import Tuple
import torch
import os
import numpy as np
from .constants import TRAIN_MODE_LACAM_ONLY, TRAIN_MODE_BEST

SEED_MAX = 2**32 - 1


def run_pipeline(args: argparse.Namespace) -> None:
    if args.training_mode == TRAIN_MODE_LACAM_ONLY:
        _run_lacam_only(args)
        return

    model, training_stats = _run_model_training(args)

    print("Training completed.")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "trained_model.pt")
        torch.save(model.state_dict(), model_path)
        json_path = os.path.join(args.output_dir, "training_stats.json")
        training_stats.save_as_json(json_path)


def _initialize_model(
    path: str | None, apply_pretraining: bool = False, grid=None
) -> DistanceTableCNN:
    if path is not None:
        return load_model(path)

    model = DistanceTableCNN()
    if apply_pretraining:
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        pretrain_on_default_value(model, grid, pretrain_optimizer, num_epochs=1000)
    return model


def _run_model_training(
    args: argparse.Namespace,
) -> Tuple[DistanceTableCNN, TrainingStats]:
    grid = get_grid(args.map_file)
    starts, goals = get_scenario(args.scen_file, args.num_agents)
    model: DistanceTableCNN | None = _initialize_model(args.model_file)
    training_stats: TrainingStats = TrainingStats()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    solution_found = False
    random_seed_gen = random.Random(args.seed)

    # Start training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        seed = random_seed_gen.randint(0, SEED_MAX)

        # solve MAPF
        planner = LaCAM()

        solution = planner.solve(
            grid=grid,
            starts=starts,
            goals=goals,
            model=model,
            seed=seed,
            time_limit_ms=args.time_limit_ms,
            flg_star=args.flg_star,
            verbose=args.verbose,
        )
        if len(solution) != 0:
            solution_found = True
            validate_mapf_solution(grid, starts, goals, solution)

        dist_tables_model = planner.dist_tables

        soc_with_model = None
        soc_without_model = None
        dist_table_difference = None

        if args.training_mode == TRAIN_MODE_BEST:
            # Solve again without model
            solution_no_model = planner.solve(
                grid=grid,
                starts=starts,
                goals=goals,
                model=None,
                seed=seed,
                time_limit_ms=args.time_limit_ms,
                flg_star=args.flg_star,
                verbose=args.verbose,
            )
            validate_mapf_solution(grid, starts, goals, solution_no_model)
            soc_without_model = get_soc(solution_no_model)

            dist_table_difference = float(
                np.mean(
                    [
                        np.abs(dt_model.table - dt_no_model.table).mean().item()
                        for dt_model, dt_no_model in zip(
                            dist_tables_model, planner.dist_tables
                        )
                    ]
                )
            )

            if not solution_found:
                solution = solution_no_model
                soc_with_model = None
            else:
                soc_with_model = get_soc(solution)
                if soc_without_model < soc_with_model:
                    solution = solution_no_model

        soc = get_soc(solution)

        # train model
        mean_loss = train_on_lacam_solution(
            model, optimizer, solution, starts, goals, grid, device_str=args.device
        )

        training_stats.record_epoch(
            mean_loss,
            soc,
            soc_model=soc_with_model,
            soc_no_model=soc_without_model,
            dist_table_diff=dist_table_difference,
        )

        print(f"  SOC: {soc}, Mean Loss: {mean_loss:.4f}")
        solution_found = False
    return model, training_stats


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
