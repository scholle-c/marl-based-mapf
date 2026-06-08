import argparse
from typing import Callable
from loguru import logger
import numpy as np

from marl_path.pycam import LaCAM
from marl_path.model import get_soc, update_from_batch
from marl_path.shared.mapf_utils import validate_mapf_solution

from .base import DefaultTrainingPipeline

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
        running_mean = None
        weight_history = []

        for epoch in range(self.args.epochs):
            logger.info(f"\n====== Epoch {epoch + 1}/{self.args.epochs} ======")
            batch = []
            weights = []
            socs_model = []
            socs_expert = []
            runtimes_model = []
            mean_delays = []
            max_delays = []

            no_solution_count = 0

            for batch_idx in range(batch_size):
                solution = None
                soc_expert = None
                soc_model = None
                seed = self.random_seed_gen.randint(0, SEED_MAX)
                no_solution_count = 0

                # Run solver with model for heuristic guidance
                model_planner = LaCAM()
                solution_model = model_planner.solve(
                    grid=self.grid,
                    starts=self.starts,
                    goals=self.goals,
                    seed=seed,
                    model=self.model,
                    device=self.device,
                    extractor=self.extractor,
                    time_limit_ms=self.args.time_limit_ms,
                    flg_star=self.args.flg_star,
                    verbose=self.args.verbose,
                )

                if len(solution_model) != 0:
                    soc_model = get_soc(solution_model)
                    validate_mapf_solution(
                        self.grid, self.starts, self.goals, solution_model
                    )
                    runtimes_model.append(model_planner.deadline.elapsed)
                else:
                    no_solution_count += 1
                    continue

                # Run solver without model to compare their socs (for reward signal)
                expert_planner = LaCAM()
                solution_expert = expert_planner.solve(
                    grid=self.grid,
                    starts=self.starts,
                    goals=self.goals,
                    seed=seed,
                    time_limit_ms=self.args.time_limit_ms,
                    flg_star=self.args.flg_star,
                    verbose=self.args.verbose,
                )

                if len(solution_expert) != 0:
                    soc_expert = get_soc(solution_expert)
                    validate_mapf_solution(
                        self.grid, self.starts, self.goals, solution_expert
                    )
                    self._record_coverage()
                else:
                    # TODO: If the expert finds no solution and the model does, its a good case --> training might be feasable!
                    continue

                # Get tensors and weight for this episode
                if soc_model is not None and soc_expert is not None:
                    solution = (
                        solution_model if soc_model < soc_expert else solution_expert
                    )
                input_tensors = [
                    dist_table.input_tensor for dist_table in model_planner.dist_tables
                ]
                bfs_tables = [
                    dist_table.table for dist_table in model_planner.dist_tables
                ]
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
                weights.append(weight)

                # Record stats for this episode
                socs_model.append(soc_model)
                socs_expert.append(soc_expert)
                delay_tables = [dt.delay for dt in model_planner.dist_tables]
                mean_delays.append(np.mean(delay_tables))
                max_delays.append(np.max(delay_tables))

            if no_solution_count > 0:
                logger.warning(
                    f"{no_solution_count} episodes in this epoch had no solution from the model."
                )

            mean_loss = update_from_batch(
                self.model, self.optimizer, batch, weights=weights
            )
            mean_soc_model = (
                sum(socs_model) / len(socs_model) if socs_model else float("nan")
            )
            mean_soc_expert = (
                sum(socs_expert) / len(socs_expert) if socs_expert else float("nan")
            )
            median_soc_model = np.median(socs_model) if socs_model else float("nan")
            median_soc_expert = np.median(socs_expert) if socs_expert else float("nan")
            std_soc_model = np.std(socs_model) if socs_model else float("nan")
            std_soc_expert = np.std(socs_expert) if socs_expert else float("nan")
            mean_runtime_model = (
                sum(runtimes_model) / len(runtimes_model)
                if runtimes_model
                else float("nan")
            )
            improvement_count = sum(1 for m, e in zip(socs_model, socs_expert) if m < e)
            improvement_percentage = (
                (improvement_count / len(socs_model) * 100)
                if socs_model
                else float("nan")
            )
            avg_weight = sum(weights) / len(weights) if weights else float("nan")
            weight_history.append(avg_weight)
            weight_history_avg = (
                sum(weight_history) / len(weight_history)
                if weight_history
                else float("nan")
            )
            mean_delay = (
                sum(mean_delays) / len(mean_delays) if mean_delays else float("nan")
            )
            mean_max_delay = (
                sum(max_delays) / len(max_delays) if max_delays else float("nan")
            )

            self.training_stats.record_epoch(
                mean_loss, mean_soc_model, mean_runtime_model
            )

            logger.info(
                "Epoch {}: mean_loss={:.4f}, mean_soc_model={:.2f}, mean_soc_expert={:.2f}, median_soc_model={:.2f}, median_soc_expert={:.2f}, std_soc_model={:.2f}, std_soc_expert={:.2f}, improvement_percentage={:.2f}%, avg_weight={:.2f}, weight_history_avg={:.2f}, mean_delay={:.2f}, mean_max_delay={:.2f}",
                epoch + 1,
                mean_loss,
                mean_soc_model,
                mean_soc_expert,
                median_soc_model,
                median_soc_expert,
                std_soc_model,
                std_soc_expert,
                improvement_percentage,
                avg_weight,
                weight_history_avg,
                mean_delay,
                mean_max_delay,
            )

        logger.info("Training completed after {} epochs.", self.args.epochs)
