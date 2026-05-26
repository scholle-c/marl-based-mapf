from pathlib import Path
from typing import Dict, List, Tuple
from marl_path.model import TrainingStats
import numpy as np

# Constants
BASE_DIR = "./output"
OVERVIEW_PAGE = "overview_page.py"

# Variables
cwd: Path = Path(BASE_DIR)
selected_folders: List = []
train_stats: List[TrainingStats] = []
dist_tables_model: List = []
dist_tables_lacam: List = []
map_mask: np.ndarray | None = None
agent_paths: List[List[List[Tuple[int, int]]]] = []

# Comparison page: dict mapping variant name -> list of TrainingStats (one per run)
comparison_variants: Dict[str, List[TrainingStats]] = {}
# Parallel to comparison_variants: run folder names for each variant
comparison_variant_run_names: Dict[str, List[str]] = {}
