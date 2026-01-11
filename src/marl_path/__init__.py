from .pipeline import run_pipeline
from .cli import main
import sys
from loguru import logger

# set logger
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>",
)

__all__ = ["run_pipeline", "main"]
