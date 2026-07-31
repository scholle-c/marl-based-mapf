"""Package containing the pathfinding model definition and modules for training, inference and evaluation of the model."""

from .definition import DistanceTableCNN, PatchTransformer, DefaultModel
from .training import (
    pretrain_on_default_value,
    pretrain_on_bfs,
    DenseDelayBatchItem,
    prepare_dense_delay_batch_item,
    update_dense_delay_from_batch,
    eval_dense_delay_loss,
    trivial_baseline_dense_loss,
    compute_mask_iou_f1,
    compute_cell_overlap,
    update_dense_delay_regression_from_batch,
    eval_dense_delay_regression_loss,
    compute_regression_metrics,
    trivial_baseline_regression_loss,
)
from .inference import load_model, save_checkpoint
from .evaluation import get_soc
from .stats import TrainingStats, MAPFStats
from .utils import build_input_tensor, build_random_input_tensor
from .feature_extraction import (
    FeatureExtractor,
    BasicExtractor,
    BinaryAgentsChannelExtractor,
    AggregatedAgentsChannelExtractor,
    RichAgentsChannelExtractor,
    CollisionAwareAgentsChannelExtractor,
    PathMembershipAgentsChannelExtractor,
)

__all__ = [
    "DefaultModel",
    "DistanceTableCNN",
    "PatchTransformer",
    "load_model",
    "save_checkpoint",
    "build_input_tensor",
    "build_random_input_tensor",
    "DenseDelayBatchItem",
    "prepare_dense_delay_batch_item",
    "update_dense_delay_from_batch",
    "eval_dense_delay_loss",
    "trivial_baseline_dense_loss",
    "compute_mask_iou_f1",
    "compute_cell_overlap",
    "update_dense_delay_regression_from_batch",
    "eval_dense_delay_regression_loss",
    "compute_regression_metrics",
    "trivial_baseline_regression_loss",
    "pretrain_on_default_value",
    "pretrain_on_bfs",
    "get_soc",
    "TrainingStats",
    "MAPFStats",
    "FeatureExtractor",
    "BasicExtractor",
    "BinaryAgentsChannelExtractor",
    "AggregatedAgentsChannelExtractor",
    "RichAgentsChannelExtractor",
    "CollisionAwareAgentsChannelExtractor",
    "PathMembershipAgentsChannelExtractor",
]
