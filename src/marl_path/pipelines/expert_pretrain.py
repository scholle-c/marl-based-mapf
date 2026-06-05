import argparse
from typing import Callable
from loguru import logger

from marl_path.pycam import LaCAM
from marl_path.model import get_soc, update_from_batch
from marl_path.shared.mapf_utils import validate_mapf_solution

from .base import DefaultTrainingPipeline, _record_empty_epoch

SEED_MAX = 2**32 - 1


class ExpertAlgorithmPretrainingPipeline(DefaultTrainingPipeline):
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

        for epoch in range(self.args.epochs):
            logger.info(f"\n====== Epoch {epoch + 1}/{self.args.epochs} ======")
            batch = []
            socs_expert = []
            runtime_expert = []

            for batch_idx in range(batch_size):
                solution = None
                seed = self.random_seed_gen.randint(0, SEED_MAX)

                planner = LaCAM()
                solution = planner.solve(
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
                    socs_expert.append(soc_expert)
                    runtime_expert.append(planner.deadline.elapsed)
                    self._record_coverage()
                else:
                    continue

                values_q_tot, target_q_tot = self.compute_tensors(
                    self.model,
                    solution,
                    self.starts,
                    self.goals,
                    self.grid,
                    device=self.device,
                    extractor=self.extractor,
                )
                batch.append((values_q_tot, target_q_tot))

            mean_loss = update_from_batch(self.model, self.optimizer, batch)

            planner = LaCAM()
            solution = planner.solve(
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
            elapsed_time_model = planner.deadline.elapsed

            if len(solution) != 0:
                soc_model = get_soc(solution)
                validate_mapf_solution(self.grid, self.starts, self.goals, solution)
                self.training_stats.record_epoch(
                    mean_loss, soc_model, elapsed_time_model
                )
            else:
                soc_model = None
                _record_empty_epoch(self.training_stats)
                logger.info("No model solution found this epoch.")

            num_solved = len(socs_expert)
            mean_soc_expert = sum(socs_expert) / num_solved if socs_expert else None
            logger.info(
                f"Solving Rate Expert: {num_solved}/{batch_size}, SOC (Model): {soc_model}, Mean SOC (Expert): {mean_soc_expert}, Mean Loss: {mean_loss:.4f}"
            )

        logger.info("Training completed after {} epochs.", self.args.epochs)
