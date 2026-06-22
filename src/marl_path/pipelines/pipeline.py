import argparse
import os
from pathlib import Path
from loguru import logger

import marl_path.constants as consts
from marl_path.model import prepare_delay_batch_item

from .comparison import ComparisonPipeline
from .delay_vs_expert import DelayVsExpertPipeline

_TRAINING_MODE_FNS = {consts.TRAINING_MODE_DELAY: prepare_delay_batch_item}


def run_pipeline(args: argparse.Namespace) -> None:
    if args.output_dir is not None and args.record_mode != 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = Path(args.output_dir) / "logs_{time:YYYY-MM-DD_HH-mm-ss}.log"
        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        )

    compute_tensors = _TRAINING_MODE_FNS.get(args.training_mode)
    if compute_tensors is None:
        raise ValueError(
            f"Unknown training mode '{args.training_mode}'. "
            f"Choose from: {list(_TRAINING_MODE_FNS)}"
        )

    logger.info("MARL-path pipeline started with arguments: {}", args)
    logger.info("starting MARL-path pipeline in mode: {}", args.pipeline_mode)
    if args.pipeline_mode == consts.PIPELINE_MODE_LACAM_ONLY:
        pipeline = ComparisonPipeline(args, compute_tensors)
    else:
        pipeline = DelayVsExpertPipeline(args, compute_tensors)

    pipeline.run_model_training()
    pipeline.store_results()
