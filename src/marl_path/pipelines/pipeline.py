import argparse
import os
from pathlib import Path
from loguru import logger

import marl_path.constants as consts

from .comparison import ComparisonPipeline
from .supervised_delay import SupervisedDelayPipeline
from .supervised_delay_regression import SupervisedDelayRegressionPipeline
from .eval_only import EvalOnlyPipeline


def run_pipeline(args: argparse.Namespace) -> None:
    if args.output_dir is not None and args.record_mode != 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = Path(args.output_dir) / "logs_{time:YYYY-MM-DD_HH-mm-ss}.log"
        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        )

    logger.info("MARL-path pipeline started with arguments: {}", args)
    logger.info("Pipeline mode: {}", args.pipeline_mode)

    if args.pipeline_mode == consts.PIPELINE_MODE_LACAM_ONLY:
        pipeline = ComparisonPipeline(args)
    elif args.pipeline_mode == consts.PIPELINE_MODE_SUPERVISED_DELAY:
        pipeline = SupervisedDelayPipeline(args)
    elif args.pipeline_mode == consts.PIPELINE_MODE_SUPERVISED_DELAY_REGRESSION:
        pipeline = SupervisedDelayRegressionPipeline(args)
    elif args.pipeline_mode == consts.PIPELINE_MODE_EVAL_ONLY:
        pipeline = EvalOnlyPipeline(args)
    else:
        raise ValueError(
            f"Unknown pipeline mode '{args.pipeline_mode}'. "
            f"Choose from: {consts.PIPELINE_MODE_SUPERVISED_DELAY}, "
            f"{consts.PIPELINE_MODE_SUPERVISED_DELAY_REGRESSION}, "
            f"{consts.PIPELINE_MODE_LACAM_ONLY}, {consts.PIPELINE_MODE_EVAL_ONLY}"
        )

    pipeline.run_model_training()
    pipeline.store_results()
