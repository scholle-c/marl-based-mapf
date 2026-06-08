import argparse
import random
import os
import json
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Callable
from loguru import logger

import marl_path.constants as consts
from marl_path.model import (
    DefaultModel,
    DistanceTableCNN,
    load_model,
    pretrain_on_default_value,
    pretrain_on_bfs,
    TrainingStats,
    MAPFStats,
    BasicExtractor,
    FeatureExtractor,
    BinaryAgentsChannelExtractor,
    AggregatedAgentsChannelExtractor,
)
from marl_path.model.inference import save_checkpoint
from marl_path.shared.mapf_utils import get_grid, get_scenario


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


class DefaultTrainingPipeline(DefaultPipeline):
    """
    A default implementation of the training pipeline that can be used as a base for specific training approaches.

    This class provides a structure for loading the MAPF instance, initializing the model, and defining the interface for running the training loop and storing results. Subclasses can override the run_model_training and store_results methods to implement specific training logic and result handling.
    """

    def __init__(self, args: argparse.Namespace, compute_tensors: Callable):
        super().__init__(args)
        self.compute_tensors = compute_tensors
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
        self._start_coverage = np.zeros(self.grid.shape, dtype=np.int32)
        self._goal_coverage = np.zeros(self.grid.shape, dtype=np.int32)

    def _record_coverage(self) -> None:
        for s, g in zip(self.starts.positions, self.goals.positions):
            self._start_coverage[s] += 1
            self._goal_coverage[g] += 1

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
            np.savetxt(
                os.path.join(
                    self.args.output_dir, consts.DEFAULT_FILENAME_START_COVERAGE
                ),
                self._start_coverage,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                os.path.join(
                    self.args.output_dir, consts.DEFAULT_FILENAME_GOAL_COVERAGE
                ),
                self._goal_coverage,
                delimiter=",",
                fmt="%d",
            )
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

    if extractor_type == consts.EXTRACTOR_BINARY_AGENTS_CHANNEL:
        extractor = BinaryAgentsChannelExtractor()
    elif extractor_type == consts.EXTRACTOR_AGGREGATED_AGENTS_CHANNEL:
        extractor = AggregatedAgentsChannelExtractor()
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
    elif model_initialization_mode == 2:
        logger.info("applying pretraining on BFS distance tables...")
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        pretrain_on_bfs(
            model,
            grid,
            pretrain_optimizer,
            num_epochs=1000,
            device=device,
            extractor=extractor,
        )
        logger.info("pretraining completed.")
    return model, extractor


def _get_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    if device_str == "auto":
        if has_cuda:
            return torch.device("cuda")
        elif has_mps:
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if device_str.startswith("cuda"):
        if has_cuda:
            return torch.device(device_str)
        else:
            print("CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")

    if device_str == "mps":
        if has_mps:
            return torch.device("mps")
        else:
            print("MPS is not available. Falling back to CPU.")
            return torch.device("cpu")

    raise ValueError(f"Unknown device string: {device_str}")


def _record_empty_epoch(
    training_stats: TrainingStats,
    dist_tables_model: list | None = None,
) -> None:
    logger.info("No solution found this epoch.")
    training_stats.record_epoch(loss=None, soc=None)
    if dist_tables_model is not None:
        training_stats.record_dist_tables(model_tables=dist_tables_model)
