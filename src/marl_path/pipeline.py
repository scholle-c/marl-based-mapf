import argparse
import random
from .model import (
    DefaultModel,
    DistanceTableCNN,
    load_model,
    train_vdn_on_solution,
    get_soc,
    pretrain_on_default_value,
    TrainingStats,
    MAPFStats,
    BasicExtractor,
    FeatureExtractor,
    OtherAgentsChannelExtractor,
)
from .model.inference import save_checkpoint
from marl_path.shared.mapf_utils import get_grid, get_scenario, validate_mapf_solution
from .pycam import LaCAM
import torch
import os
import marl_path.constants as consts
from loguru import logger
import numpy as np
import json
from pathlib import Path
from abc import ABC, abstractmethod

SEED_MAX = 2**32 - 1


def run_pipeline(args: argparse.Namespace) -> None:
    if args.output_dir is not None and args.record_mode != 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = Path(args.output_dir) / "logs_{time:YYYY-MM-DD_HH-mm-ss}.log"
        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        )

    logger.info("MARL-path pipeline started with arguments: {}", args)
    logger.info("starting MARL-path pipeline in mode: {}", args.pipeline_mode)
    if args.pipeline_mode == consts.TRAIN_MODE_LACAM_ONLY:
        _run_lacam_only(args)
        return

    pipeline = VDNPipeline(args)
    pipeline.run_model_training()
    pipeline.store_results()


class DefaultPipeline(ABC):
    """
    Abstract base class for the training pipeline. Defines the interface for running the pipeline and storing results.

        Subclasses should implement the run_model_training and store_results methods to define specific training and evaluation logic.
        Results of the pipeline: trained model for a heuristic distance prediction and training statistics
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model: DefaultModel
        self.training_stats: TrainingStats
        self._load_mapf_instance()

    def _load_mapf_instance(self) -> None:
        self.grid = get_grid(self.args.map_file)
        self.starts, self.goals = get_scenario(
            self.args.scen_file, self.args.num_agents
        )

    @abstractmethod
    def run_model_training(self) -> None:
        pass

    @abstractmethod
    def store_results(self) -> None:
        pass


class VDNPipeline(DefaultPipeline):
    """
    Implementation of the training pipeline using a Value Decomposition Network (VDN) approach for training a distance table CNN model based on LaCAM solutions.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.device: torch.device = _get_device(self.args.device)
        self.model, self.extractor = _initialize_model(
            self.args.model_file,
            self.device,
            model_initialization_mode=self.args.model_initialization_mode,
            grid=self.grid,
            seed=getattr(self.args, "seed_training", None),
            extractor_type=self.args.feature_extractor_type,
        )
        self.training_stats = TrainingStats(
            training_mode=self.args.pipeline_mode,
            used_device=self.device.type,
            used_seed=getattr(self.args, "seed_training", None),
            mapf=MAPFStats(map_size=self.grid.shape, num_agents=self.args.num_agents),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.random_seed_gen = random.Random(self.args.seed)

    def run_model_training(self) -> None:
        logger.info(
            "starting training loop with parameters: epochs={}, lr={}, device={}, seed={}",
            self.args.epochs,
            self.args.lr,
            self.device.type,
            self.args.seed,
        )

        # Start training loop
        for epoch in range(self.args.epochs):
            logger.info(f"\n====== Epoch {epoch + 1}/{self.args.epochs} ======")
            solution = None
            soc = None
            seed = self.random_seed_gen.randint(0, SEED_MAX)

            # solve MAPF using your model
            planner = LaCAM()

            solution = planner.solve(
                grid=self.grid,
                starts=self.starts,
                goals=self.goals,
                model=self.model,
                device=self.device,
                extractor=self.extractor,
                seed=seed,
                time_limit_ms=self.args.time_limit_ms,
                flg_star=self.args.flg_star,
                verbose=self.args.verbose,
            )

            elapsed_time = planner.deadline.elapsed

            if len(solution) != 0:
                validate_mapf_solution(self.grid, self.starts, self.goals, solution)
                soc = get_soc(solution)
            else:
                dist_tables_model = [dt.table for dt in planner.dist_tables]
                _record_empty_epoch(
                    self.training_stats,
                    dist_tables_model=dist_tables_model,
                )
                logger.info("No solution found this epoch.")
                continue

            # train model
            mean_loss = train_vdn_on_solution(
                self.model,
                self.optimizer,
                solution,
                self.starts,
                self.goals,
                self.grid,
                device=self.device,
                extractor=self.extractor,
            )

            self.training_stats.record_epoch(mean_loss, soc, elapsed_time)

            logger.info(f"  SOC: {soc}, Mean Loss: {mean_loss:.4f}")

        logger.info("Training completed after {} epochs.", self.args.epochs)

    def store_results(self) -> None:
        if self.args.record_mode != 0:
            os.makedirs(self.args.output_dir, exist_ok=True)
            # Model saving
            model_path = os.path.join(
                self.args.output_dir, consts.DEFAULT_FILENAME_TRAINED_MODEL
            )
            save_checkpoint(self.model, self.extractor, model_path)
            # Training stats saving
            map_mask = get_grid(self.args.map_file)
            self.training_stats.save(self.args.output_dir, map_mask=map_mask)
            # Arguments saving
            args_path = os.path.join(
                self.args.output_dir, consts.DEFAULT_FILENAME_USED_CONFIG
            )
            data = {
                k: (str(v) if hasattr(v, "__fspath__") else v)
                for k, v in vars(self.args).items()
            }
            with open(args_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(
                "Saved trained model and training stats to {}", self.args.output_dir
            )


def _initialize_model(
    path: str | None,
    device: torch.device,
    model_initialization_mode: int = 0,
    grid=None,
    seed: int | None = None,
    extractor_type: str = consts.EXTRACTOR_BASIC,
) -> tuple[DefaultModel, FeatureExtractor]:
    if path is not None:
        return load_model(path, device=device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    if extractor_type == consts.EXTRACTOR_OTHER_AGENTS_CHANNEL:
        extractor = OtherAgentsChannelExtractor()
    else:
        extractor = BasicExtractor()

    model = DistanceTableCNN(in_channels=extractor.n_channels).to(device)
    if model_initialization_mode == 1:
        logger.info("applying pretraining on default values...")
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        pretrain_on_default_value(
            model,
            grid,
            pretrain_optimizer,
            num_epochs=1000,
            device=device,
            extractor=extractor,
        )
        logger.info("pretraining completed.")
    return model, extractor


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


def _record_empty_epoch(
    training_stats: TrainingStats,
    dist_tables_model: list | None = None,
) -> None:
    logger.info("No solution found this epoch.")
    training_stats.record_epoch(loss=None, soc=None)
    if dist_tables_model is not None:
        training_stats.record_dist_tables(model_tables=dist_tables_model)
