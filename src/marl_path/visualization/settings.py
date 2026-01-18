from pathlib import Path
from typing import List
from marl_path.model import TrainingStats

# Constants
BASE_DIR = "./output"
OVERVIEW_PAGE = "overview_page.py"

# Variables
cwd: Path = Path(BASE_DIR)
selected_folders: List = []
train_stats: List[TrainingStats] = []
dist_tables_model: List = []
dist_tables_lacam: List = []
