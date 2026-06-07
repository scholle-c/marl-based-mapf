import argparse
from typing import Callable
from loguru import logger
import numpy as np

from marl_path.pycam import LaCAM
from marl_path.model import get_soc, update_from_batch
from marl_path.shared.mapf_utils import validate_mapf_solution

from .base import DefaultTrainingPipeline, _record_empty_epoch

SEED_MAX = 2**32 - 1


class DelayVsExpertPipeline(DefaultTrainingPipeline):
    """
    Implementation of the training pipeline using an expert algorithm (e.g., LaCAM) to generate training data for pretraining a distance table CNN model.

    This pipeline runs the expert algorithm on the given MAPF instance to obtain solutions and corresponding distance tables, which are then used to pretrain the model before any reinforcement learning training.
    """

    def __init__(self, args: argparse.Namespace, compute_tensors: Callable):
        super().__init__(args, compute_tensors)

    def run_model_training(self) -> None:
        logger.info(
            "starting training loop with parameters: epochs={}, lr={}, device={}, seed={}",
            self.args.epochs,
            self.args.lr,
            self.device.type,
            self.args.seed,
        )
        batch_size = self.args.batch_size
        alpha = 0.05  # for running mean of expert soc, used in weight computation
        min_weight = 0.1  # floor to avoid zero-gradient cold-start
        running_mean = None

        for epoch in range(self.args.epochs):
            logger.info(f"\n====== Epoch {epoch + 1}/{self.args.epochs} ======")
            batch = []
            weights = []
            socs_model = []
            socs_expert = []

            for batch_idx in range(batch_size):
                solution = None
                soc_expert = None
                soc_model = None
                seed = self.random_seed_gen.randint(0, SEED_MAX)

                # Run solver with model for heuristic guidance
                model_planner = LaCAM()
                solution = model_planner.solve(
                    grid=self.grid,
                    starts=self.starts,
                    goals=self.goals,
                    model=self.model,
                    device=self.device,
                    extractor=self.extractor,
                    time_limit_ms=self.args.time_limit_ms,
                    flg_star=self.args.flg_star,
                    verbose=self.args.verbose,
                )

                if len(solution) != 0:
                    soc_model = get_soc(solution)
                    validate_mapf_solution(self.grid, self.starts, self.goals, solution)
                else:
                    _record_empty_epoch(self.training_stats)
                    logger.info("No model solution found this epoch.")
                    continue

                # Run solver without model to compare their socs (for reward signal)
                expert_planner = LaCAM()
                solution = expert_planner.solve(
                    grid=self.grid,
                    starts=self.starts,
                    goals=self.goals,
                    seed=seed,
                    time_limit_ms=self.args.time_limit_ms,
                    flg_star=self.args.flg_star,
                    verbose=self.args.verbose,
                )

                if len(solution) != 0:
                    soc_expert = get_soc(solution)
                    validate_mapf_solution(self.grid, self.starts, self.goals, solution)
                    self._record_coverage()
                else:
                    # TODO: If the expert finds no solution and the model does, its a good case --> training might be feasable!
                    continue

                # Get tensors and weight for this episode
                input_tensors = [dist_table.input_tensor for dist_table in model_planner.dist_tables]
                bfs_tables = [dist_table.table for dist_table in model_planner.dist_tables]
                values_delay, target_delay = self.compute_tensors(
                    self.model,
                    solution,
                    self.starts,
                    device=self.device,
                    bfs_tables=bfs_tables,
                    input_tensors=input_tensors,
                )
                batch.append((values_delay, target_delay))

                
                if running_mean is None:
                    running_mean = soc_expert
                else:
                    running_mean = alpha * soc_expert + (1 - alpha) * running_mean

                raw_weight = (running_mean - soc_model) / running_mean
                weight = float(np.clip(raw_weight, -1.0, 1.0))
                if abs(weight) < min_weight:
                    weight = min_weight
                weights.append(weight)

                # Record stats for this episode
                socs_model.append(soc_model)
                socs_expert.append(soc_expert)
            
            mean_loss = update_from_batch(self.model, self.optimizer, batch, weights=weights)
            mean_soc_model = sum(socs_model) / len(socs_model) if socs_model else float("nan")
            mean_soc_expert = sum(socs_expert) / len(socs_expert) if socs_expert else float("nan")


            logger.info(
                "Epoch {}: mean_loss={:.4f}, mean_soc_model={:.2f}, mean_soc_expert={:.2f}",
                epoch + 1,
                mean_loss,
                mean_soc_model,
                mean_soc_expert
            )

        logger.info("Training completed after {} epochs.", self.args.epochs)
