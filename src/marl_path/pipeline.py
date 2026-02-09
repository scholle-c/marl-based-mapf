import argparse
import random
from .model import (
    DistanceTableCNN,
    load_model,
    train_on_lacam_solution,
    get_soc,
    pretrain_on_default_value,
    TrainingStats,
    get_epsilon_sine,
)
from marl_path.shared.mapf_utils import get_grid, get_scenario, validate_mapf_solution
from .pycam import LaCAM
from typing import Tuple
import torch
import os
import marl_path.constants as consts
from loguru import logger
import numpy as np
import json

SEED_MAX = 2**32 - 1


def run_pipeline(args: argparse.Namespace) -> None:
    logger.info("starting MARL-path pipeline in mode: {}", args.training_mode)
    if args.training_mode == consts.TRAIN_MODE_LACAM_ONLY:
        _run_lacam_only(args)
        return

    model, training_stats, solutions = _run_model_training(args)

    logger.info("training completed.")
    if args.training_mode == consts.TRAIN_MODE_BEST:
        logger.info(
            "Successful model epochs: {} out of {} epochs ({:.2f}%)",
            training_stats.successful_model_epochs,
            training_stats.epochs,
            (training_stats.successful_model_epochs / training_stats.epochs * 100)
            if training_stats.epochs > 0
            else 0,
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        # Model saving
        model_path = os.path.join(
            args.output_dir, consts.DEFAULT_FILENAME_TRAINED_MODEL
        )
        torch.save(model.state_dict(), model_path)
        # Training stats saving
        map_mask = get_grid(args.map_file) if args.save_map_mask else None
        training_stats.save(args.output_dir, map_mask=map_mask)
        # Arguments saving
        args_path = os.path.join(args.output_dir, consts.DEFAULT_FILENAME_USED_CONFIG)
        data = {
            k: (str(v) if hasattr(v, "__fspath__") else v)
            for k, v in vars(args).items()
        }
        with open(args_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info("Saved trained model and training stats to {}", args.output_dir)
        # Agent paths saving
        if len(solutions) > 0:
            agent_paths_path = os.path.join(
                args.output_dir, consts.DEFAULT_FILENAME_AGENT_PATHS
            )
            with open(agent_paths_path, "w") as f:
                json.dump(solutions, f, indent=4)
            logger.info("Saved agent paths to {}", agent_paths_path)


def _initialize_model(
    path: str | None,
    device: torch.device,
    apply_pretraining: bool = False,
    grid=None,
    seed: int | None = None,
) -> DistanceTableCNN:
    if path is not None:
        return load_model(path, device=device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    model = DistanceTableCNN().to(device)
    if apply_pretraining:
        logger.info("applying pretraining on default values...")
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        pretrain_on_default_value(
            model, grid, pretrain_optimizer, num_epochs=1000, device=device
        )
        logger.info("pretraining completed.")
    return model


def _run_model_training(
    args: argparse.Namespace,
) -> Tuple[DistanceTableCNN, TrainingStats, list]:
    grid = get_grid(args.map_file)
    solutions: list = []
    starts, goals = get_scenario(args.scen_file, args.num_agents)
    device: torch.device = _get_device(args.device)
    model: DistanceTableCNN | None = _initialize_model(
        args.model_file,
        device,
        apply_pretraining=args.use_pretraining,
        grid=grid,
        seed=getattr(args, "seed_training", None),
    )
    training_stats: TrainingStats = TrainingStats(
        dist_table_record_mode=args.dist_table_record_mode,
        training_mode=args.training_mode,
        used_device=device.type,
        used_seed=getattr(args, "seed_training", None),
        map_size=grid.shape,
        num_agents=args.num_agents,
        dist_table_record_granularity=args.dist_table_record_granularity,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    solution_found_model = False
    random_seed_gen = random.Random(args.seed)
    random_epsilon_gen = random.Random(args.seed)

    logger.info(
        "starting training loop with parameters: epochs={}, lr={}, device={}, seed={}",
        args.epochs,
        args.lr,
        device.type,
        args.seed,
    )

    soc_with_model = None
    soc_without_model = None
    solution = None

    # Start training loop
    for epoch in range(args.epochs):
        logger.info(f"\n====== Epoch {epoch + 1}/{args.epochs} ======")
        seed = random_seed_gen.randint(0, SEED_MAX)

        # solve MAPF using your model
        planner = LaCAM()

        solution_model = planner.solve(
            grid=grid,
            starts=starts,
            goals=goals,
            model=model,
            device=device,
            seed=seed,
            time_limit_ms=args.time_limit_ms,
            flg_star=args.flg_star,
            verbose=args.verbose,
        )
        if len(solution_model) != 0:
            solution_found_model = True
            validate_mapf_solution(grid, starts, goals, solution_model)
            soc_with_model = get_soc(solution_model)

        dist_tables_model = [dt.table for dt in planner.dist_tables]
        dist_tables_lacam = None
        solution = solution_model

        if args.training_mode == consts.TRAIN_MODE_BEST:
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
            solution_found_lacam = len(solution_no_model) != 0
            if solution_found_lacam:
                validate_mapf_solution(grid, starts, goals, solution_no_model)
                soc_without_model = get_soc(solution_no_model)
                dist_tables_lacam = [dt.table for dt in planner.dist_tables]

            if not solution_found_model and not solution_found_lacam:
                _record_empty_epoch(
                    training_stats,
                    dist_tables_lacam=dist_tables_lacam,
                    dist_tables_model=dist_tables_model,
                )
                continue

            # TODO: Kannst in eine eigene Methode auslagern, sowas wie "determine_best_solution"
            # Check which solution is better
            better_solution = None
            worse_solution = None

            if soc_without_model and soc_with_model:
                if soc_without_model < soc_with_model:
                    better_solution = solution_no_model
                    worse_solution = solution_model
                    logger.opt(colors=True).info(
                        "Best solution comes from: <red>no model</red>"
                    )
                else:
                    better_solution = solution_model
                    worse_solution = solution_no_model
                    logger.opt(colors=True).info(
                        "Best solution comes from: <green>with model</green>"
                    )
            elif soc_without_model is not None:
                better_solution = solution_no_model
                logger.opt(colors=True).info(
                    "Best solution comes from: <red>no model</red>"
                )
            else:
                better_solution = solution_model
                logger.opt(colors=True).info(
                    "Best solution comes from: <green>with model</green>"
                )

            # Check for epsilon-greedy exploration
            # TODO: Gerne auch in eine eigene Methode auslagern
            if args.epsilon_function == consts.EPSILON_FUNCTION_FIXED:
                epoch_fraction = epoch / args.epochs
                if epoch_fraction < 0.05 or epoch_fraction > 0.95:
                    epsilon = args.epsilon_min  # exploitation
                else:
                    epsilon = args.epsilon_max  # exploration
            elif args.epsilon_function == consts.EPSILON_FUNCTION_SINE:
                epsilon = get_epsilon_sine(
                    epoch,
                    args.epochs,
                    args.epsilon_min,
                    args.epsilon_max,
                )
            else:
                epsilon = 0.0  # no exploration
            random_value = random_epsilon_gen.random()

            if random_value < epsilon and worse_solution is not None:
                logger.opt(colors=True).info(
                    "Exploration: <yellow>Using worse solution due to epsilon-greedy ({:.4f} < {:.4f})</yellow>".format(
                        random_value, epsilon
                    )
                )
                solution = worse_solution
            else:
                solution = better_solution

        # Train model only if a solution was found
        if len(solution) == 0:
            _record_empty_epoch(
                training_stats,
                dist_tables_lacam=dist_tables_lacam,
                dist_tables_model=dist_tables_model,
            )
            continue

        soc = get_soc(solution)
        _record_solution(args, solution, solutions, epoch)

        # train model
        mean_loss = train_on_lacam_solution(
            model,
            optimizer,
            solution,
            starts,
            goals,
            grid,
            device=device,
            use_neighbors=args.use_neighbors,
            goal_weight=args.goal_weight,
            use_bellman_loss=args.use_bellman_loss,
        )

        training_stats.record_epoch(
            mean_loss,
            soc,
            soc_model=soc_with_model,
            soc_no_model=soc_without_model,
            dist_tables_lacam=dist_tables_lacam,
            dist_tables_model=dist_tables_model,
        )

        logger.info(f"  SOC: {soc}, Mean Loss: {mean_loss:.4f}")
        solution_found_model = False
    return model, training_stats, solutions


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


def _get_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")

    has_cuda = torch.cuda.is_available()

    if device_str == "auto":
        if has_cuda:
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            print("CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device string: {device_str}")


def _record_solution(
    args: argparse.Namespace, solution, solutions: list, epoch: int
) -> None:
    if args.agent_path_record_mode == consts.AGENT_PATH_RECORD_MODE_NONE:
        return
    if epoch % args.agent_path_record_granularity != 0:
        return
    temp = []
    if args.agent_path_record_mode == consts.AGENT_PATH_RECORD_MODE_ALL:
        for conf in solution:
            temp.append(conf.positions)
        solutions.append(temp)
    elif args.agent_path_record_mode == consts.AGENT_PATH_RECORD_MODE_ONE_AGENT:
        for conf in solution:
            temp.append([conf.positions[0]])
        solutions.append(temp)


def _record_empty_epoch(
    training_stats: TrainingStats,
    dist_tables_lacam: list | None,
    dist_tables_model: list | None,
) -> None:
    logger.info("No solution found this epoch.")
    training_stats.record_epoch(
        train_loss=None,
        soc=None,
        soc_model=None,
        soc_no_model=None,
        dist_tables_lacam=dist_tables_lacam,
        dist_tables_model=dist_tables_model,
    )
