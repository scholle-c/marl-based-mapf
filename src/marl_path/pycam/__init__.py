import sys

from loguru import logger

from .lacam import LaCAM

# set logger
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>",
)

__all__ = [
    "LaCAM",
]
