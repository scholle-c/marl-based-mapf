import argparse
import json
import os

import numpy as np
import torch
from abc import ABC, abstractmethod
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
    RichAgentsChannelExtractor,
    CollisionAwareAgentsChannelExtractor,
    PathMembershipAgentsChannelExtractor,
)
from marl_path.model.inference import save_checkpoint
from marl_path.shared.mapf_utils import get_grid, get_scenario


class DefaultPipeline(ABC):
    """Abstract base: loads a MAPF instance and defines the pipeline interface."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model: DefaultModel
        self.training_stats: TrainingStats
        self._load_mapf_instance()

    def _load_mapf_instance(self) -> None:
        map_file = getattr(self.args, "map_file", None)
        scen_file = getattr(self.args, "scen_file", None)
        if map_file is None or scen_file is None:
            self.grid = self.starts = self.goals = None
            return
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
    """Base for training pipelines: initializes model, extractor, optimizer, and stats."""

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
            hidden_channels=getattr(self.args, "hidden_channels", 32),
            depth=getattr(self.args, "depth", 4),
        )
        self.training_stats = TrainingStats(
            training_mode=self.args.pipeline_mode,
            used_device=self.device.type,
            used_seed=getattr(self.args, "seed_training", None),
            mapf=MAPFStats(
                map_size=self.grid.shape if self.grid is not None else None,
                num_agents=self.args.num_agents,
            ),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )

    def store_results(self) -> None:
        if self.args.record_mode != 0:
            os.makedirs(self.args.output_dir, exist_ok=True)
            model_path = os.path.join(
                self.args.output_dir, consts.DEFAULT_FILENAME_TRAINED_MODEL
            )
            save_checkpoint(self.model, self.extractor, model_path)
            map_mask = (
                get_grid(self.args.map_file)
                if getattr(self.args, "map_file", None)
                else None
            )
            self.training_stats.save(self.args.output_dir, map_mask=map_mask)
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
    hidden_channels: int = 32,
    depth: int = 4,
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
    elif extractor_type == consts.EXTRACTOR_RICH_AGENTS_CHANNEL:
        extractor = RichAgentsChannelExtractor()
    elif extractor_type == consts.EXTRACTOR_COLLISION_AWARE:
        extractor = CollisionAwareAgentsChannelExtractor()
    elif extractor_type == consts.EXTRACTOR_PATH_ALL_AGENTS:
        extractor = PathMembershipAgentsChannelExtractor(agents_filter="all")
    elif extractor_type == consts.EXTRACTOR_PATH_COLLIDING_AGENTS:
        extractor = PathMembershipAgentsChannelExtractor(agents_filter="colliding")
    else:
        extractor = BasicExtractor()

    model = DistanceTableCNN(
        in_channels=extractor.n_channels,
        hidden_channels=hidden_channels,
        depth=depth,
    ).to(device)
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
