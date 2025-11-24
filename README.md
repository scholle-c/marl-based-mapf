# Value Map Learner

This repository currently contains the base scaffolding for the Value Map Learner project.

## Development Environment

- **Python:** 1.13.2 (use a virtual environment, e.g., `.venv`)
- **Package manager:** `pip`

## Setup

1. Install Python 1.13.2.
2. Create a virtual environment:
   - `python -m venv .venv`
3. Activate the environment:
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
   - bash: `source .venv/bin/activate`
4. Install dependencies:
   - `pip install -r requirements.txt`

## Installed Packages

The virtual environment currently includes the libraries installed via `pip3 install torch torchvision`:

- `torch`
- `torchvision`

Add additional dependencies to `requirements.txt` as the project evolves.

## Usage

### CLI overview

The CLI now uses subcommands. `app.py` remains as a thin wrapper that forwards to this interface.

- Generate archives from maps:
  - `python -m value_map_learner.cli generate --maps assets/connector.map assets/corners.map --output-dir data/generated --samples-per-map 200`
- Train from existing archives:
  - `python -m value_map_learner.cli train --train-data data/train_*.npz --output-dir output`
- Train while generating in-memory (no disk writes):
  - `python -m value_map_learner.cli train --maps assets/connector.map assets/corners.map --samples-per-map 200 --epochs 20`
- Train while generating and persisting the generated data:
  - `python -m value_map_learner.cli train --maps assets/connector.map --save-generated-data data/generated --output-dir output`

### Config-driven training

You can point the `train` subcommand at a config file (JSON/TOML/YAML). CLI flags override config values when both are present.

Example `configs/train.json`:

```json
{
  "train_archives": ["data/train_001.npz", "data/train_002.npz"],
  "val_archives": ["data/val_001.npz"],
  "epochs": 15,
  "batch_size": 8,
  "learning_rate": 0.001,
  "device": "cuda",
  "output_dir": "output"
}
```

Run with:

```python
python -m value_map_learner.cli train --config configs/train.json
```

### Config-driven data generation

Example `configs/data_generation.json`:

```json
{
  "maps": ["assets/connector.map", "assets/corners.map"],
  "samples_per_map": 200,
  "output_dir": "data/generated"
}
```

Generate archives using the config (CLI flags still override config keys):

```python
python -m value_map_learner.cli generate --config configs/data_generation.json
```

## Testing

- Install test dependencies: `pip install -r requirements-dev.txt`
- Run the suite: `pytest`

### Training + Plotting Workflow

1. Train and save outputs (model + loss history):
   - `python -m value_map_learner.cli train --train-data data/train_*.npz --output-dir output`
   - Outputs written to `output/model.pt` and `output/loss_history.json`.
2. Plot loss curve:
   - `python evaluation_utils.py`
   - Outputs `output/loss_curve.png` (override paths in `evaluation_utils.py` if needed).
3. Plot a predicted value map for a goal on a given map:
   - In Python:

     ```python
     from evaluation_utils import plot_predicted_value_map
     plot_predicted_value_map("path/to/map.map", goal=(y, x), model_path="output/model.pt", output_path="output/pred_value_map.png")
     ```
