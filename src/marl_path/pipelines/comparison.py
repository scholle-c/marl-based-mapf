import argparse
from loguru import logger

from marl_path.pycam import LaCAM
from marl_path.model import get_soc
from marl_path.shared.mapf_utils import validate_mapf_solution

from .base import DefaultTrainingPipeline

SEED_MAX = 2**32 - 1


class ComparisonPipeline(DefaultTrainingPipeline):
    """Runs LaCAM without a trained model to produce a SOC baseline for comparison."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        if self.grid is None or self.starts is None or self.goals is None:
            raise ValueError("ComparisonPipeline requires --map-file and --scen-file.")

    def run_model_training(self) -> None:
        assert (
            self.grid is not None and self.starts is not None and self.goals is not None
        )
        logger.info(
            "starting comparison run: epochs={}, device={}, seed={}",
            self.args.epochs,
            self.device.type,
            self.args.seed,
        )
        import random

        seed_gen = random.Random(self.args.seed)

        for epoch in range(self.args.epochs):
            logger.info(f"\n====== Epoch {epoch + 1}/{self.args.epochs} ======")
            seed = seed_gen.randint(0, SEED_MAX)

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

            if solution:
                soc = get_soc(solution)
                validate_mapf_solution(self.grid, self.starts, self.goals, solution)
                self.training_stats.record_epoch(None, soc, planner.deadline.elapsed)
                logger.info(f"SOC: {soc}")
            else:
                self.training_stats.record_epoch(None, None, None)
                logger.info("No solution found this epoch.")

        logger.info("Comparison run completed after {} epochs.", self.args.epochs)
