"""
Contains constants for the marl-path package.
"""

TRAIN_MODE_MODEL: str = "model"
TRAIN_MODE_LACAM_ONLY: str = "lacam_only"
TRAIN_MODE_BEST: str = "best"

# Filenames
DEFAULT_FILENAME_TRAINING_STATS: str = "training_stats.json"
DEFAULT_FILENAME_USED_CONFIG: str = "used_config.json"
DEFAULT_FILENAME_TRAINED_MODEL: str = "trained_model.pt"
DEFAULT_FILENAME_DIST_TABLE_MODEL: str = "dist_tables_model.csv"
DEFAULT_FILENAME_DIST_TABLE_LACAM: str = "dist_tables_lacam.csv"
DEFAULT_FILENAME_MAP_MASK: str = "map_mask.csv"
DEFAULT_FILENAME_AGENT_PATHS: str = "agent_paths.json"

# Folder Names
FOLDER_COMPARISON_ALGOS: str = "other_algos"
EXPERT_ALGO_NAME: str = "expert_algo"

COMPARISON_ALGO_NONE: str = "none"
COMPARISON_ALGO_LACAM: str = "lacam"
COMPARISON_ALGO_IMITATION_MODEL: str = "imitation_model"

# Pipeline modes
PIPELINE_MODE_VDN: str = "vdn"
PIPELINE_MODE_EXPERT_PRETRAIN: str = "expert_pretrain"
PIPELINE_MODE_LACAM_ONLY: str = "lacam_only"

# Extractor types
EXTRACTOR_BASIC: str = "basic"
EXTRACTOR_OTHER_AGENTS_CHANNEL: str = "other_agents_channel"
