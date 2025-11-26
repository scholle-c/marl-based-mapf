"""Lightweight package exposing the pathfinding CNN and helper utilities."""

from .distance_table_cnn import DistanceTableCNN, load_model

__all__ = ["DistanceTableCNN", "load_model"]
