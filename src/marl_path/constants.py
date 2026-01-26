"""
Contains constants for the marl-path package.
"""

TRAIN_MODE_MODEL: str = "model"
TRAIN_MODE_LACAM_ONLY: str = "lacam_only"
TRAIN_MODE_BEST: str = "best"

DEFAULT_FILENAME_TRAINING_STATS: str = "training_stats.json"
DEFAULT_FILENAME_USED_CONFIG: str = "used_config.json"
DEFAULT_FILENAME_TRAINED_MODEL: str = "trained_model.pt"
DEFAULT_FILENAME_DIST_TABLE_MODEL: str = "dist_tables_model.csv"
DEFAULT_FILENAME_DIST_TABLE_LACAM: str = "dist_tables_lacam.csv"
DEFAULT_FILENAME_MAP_MASK: str = "map_mask.csv"
DEFAULT_FILENAME_AGENT_PATHS: str = "agent_paths.json"

AGENT_PATH_RECORD_MODE_NONE: int = 0
AGENT_PATH_RECORD_MODE_ALL: int = 1
AGENT_PATH_RECORD_MODE_ONE_AGENT: int = 2
