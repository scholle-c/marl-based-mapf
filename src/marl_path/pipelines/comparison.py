import argparse
from typing import Callable
from loguru import logger

from marl_path.pycam import LaCAM
from marl_path.model import get_soc
from marl_path.shared.mapf_utils import validate_mapf_solution

from .base import DefaultTrainingPipeline, _record_empty_epoch

SEED_MAX = 2**32 - 1


class ComparisonPipeline(DefaultTrainingPipeline):
    """
    A pipeline that runs an algorithm (e.g., LaCAM) without any training to obtain
    results to compare with a trained model.
    """

    def __init__(self, args: argparse.Namespace, compute_tensors: Callable):
        super().__init__(args, compute_tensors)
        self.USE_LACAM = True

    def run_model_training(self) -> None:
        logger.info(
            "starting training loop with parameters: epochs={}, lr={}, device={}, seed={}",
            self.args.epochs,
            self.args.lr,
            self.device.type,
            self.args.seed,
        )

        for epoch in range(self.args.epochs):
            logger.info(f"\n====== Epoch {epoch + 1}/{self.args.epochs} ======")
            solution = None
            soc = None
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
            elapsed_time_model = planner.deadline.elapsed

            if len(solution) != 0:
                soc = get_soc(solution)
                validate_mapf_solution(self.grid, self.starts, self.goals, solution)
                self.training_stats.record_epoch(None, soc, elapsed_time_model)
            else:
                dist_tables_model = [dt.table for dt in planner.dist_tables]
                _record_empty_epoch(
                    self.training_stats,
                    dist_tables_model=dist_tables_model,
                )
                logger.info("No model solution found this epoch.")

            logger.info(f" SOC: {soc}")

        logger.info("Training completed after {} epochs.", self.args.epochs)
