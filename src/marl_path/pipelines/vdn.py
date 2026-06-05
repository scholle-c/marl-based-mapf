import argparse
from typing import Callable
from loguru import logger

from marl_path.pycam import LaCAM
from marl_path.model import get_soc, update_from_batch
from marl_path.shared.mapf_utils import validate_mapf_solution

from .base import DefaultTrainingPipeline

SEED_MAX = 2**32 - 1


class VDNPipeline(DefaultTrainingPipeline):
    """
    Implementation of the training pipeline using a Value Decomposition Network (VDN) approach for training a distance table CNN model based on LaCAM solutions.
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
            socs = []
            runtimes = []

            for batch_idx in range(batch_size):
                seed = self.random_seed_gen.randint(0, SEED_MAX)

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
                    socs.append(soc)
                    runtimes.append(elapsed_time)
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

            num_solved = len(socs)
            mean_soc = sum(socs) / len(socs) if socs else None
            mean_runtime = sum(runtimes) / len(runtimes) if runtimes else None

            self.training_stats.record_epoch(mean_loss, mean_soc, mean_runtime)

            logger.info(
                f"Solving Rate: {num_solved}/{batch_size}, Mean SOC: {mean_soc}, Mean Loss: {mean_loss:.4f}"
            )

        logger.info("Training completed after {} epochs.", self.args.epochs)
