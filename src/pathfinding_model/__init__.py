"""Lightweight package exposing the pathfinding CNN and helper utilities."""

from .distance_table_cnn import DistanceTableCNN, load_model
from .model_training import train_on_solution, get_soc, pretrain_model_on_default_value
from .model_utils import build_input_tensor, build_random_input_tensor

__all__ = ["DistanceTableCNN", "load_model", "build_input_tensor", "build_random_input_tensor", "train_on_solution", "get_soc", "pretrain_model_on_default_value"]