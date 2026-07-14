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
DEFAULT_FILENAME_START_COVERAGE: str = "start_coverage.csv"
DEFAULT_FILENAME_GOAL_COVERAGE: str = "goal_coverage.csv"

# Folder Names
FOLDER_COMPARISON_ALGOS: str = "other_algos"
EXPERT_ALGO_NAME: str = "expert_algo"

COMPARISON_ALGO_NONE: str = "none"
COMPARISON_ALGO_LACAM: str = "lacam"
COMPARISON_ALGO_IMITATION_MODEL: str = "imitation_model"

# Pipeline modes
PIPELINE_MODE_SUPERVISED_DELAY: str = "supervised_delay"
PIPELINE_MODE_LACAM_ONLY: str = "lacam_only"
PIPELINE_MODE_EVAL_ONLY: str = "eval_only"

# Extractor types
EXTRACTOR_BASIC: str = "basic"
EXTRACTOR_BINARY_AGENTS_CHANNEL: str = "binary_agents_channel"
EXTRACTOR_AGGREGATED_AGENTS_CHANNEL: str = "aggregated_agents_channel"
EXTRACTOR_RICH_AGENTS_CHANNEL: str = "rich_agents_channel"
EXTRACTOR_COLLISION_AWARE: str = "collision_aware"
EXTRACTOR_PATH_ALL_AGENTS: str = "path_all_agents"
EXTRACTOR_PATH_COLLIDING_AGENTS: str = "path_colliding_agents"
EXTRACTOR_PATH_ALL_AGENTS_INTERSECTION: str = "path_all_agents_intersection"
EXTRACTOR_PATH_ALL_AGENTS_TIME: str = "path_all_agents_time"
EXTRACTOR_PATH_ALL_AGENTS_INTERSECTION_TIME: str = "path_all_agents_intersection_time"
