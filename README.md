# MAPF-MARL-pipeline project

This repository contains an experimental environment for trying to combine MAPF (using the LaCAM algorithm) with MARL.

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

## Usage

Run the end-to-end MAPF pipeline from the project root:

```console
python app.py --config-file configs/default_config.toml
```

CLI flags override values in the TOML config. Available arguments:

| Argument | Description | Default value |
| --- | --- | --- |
| `-c, --config-file` | TOML config to load pipeline settings from. | `configs/default_config.toml` |
| `-m, --map-file` | Grid map file for the MAPF instance. | `assets/tunnel.map` |
| `-i, --scen-file` | Scenario file listing start/goal pairs. | `assets/tunnel.scen` |
| `-N, --num-agents` | Number of agents to read from the scenario. | `4` |
| `-v, --verbose` | Verbosity level for logging. | `1` |
| `-s, --seed` | Random seed for the LaCAM planner. | `0` |
| `-t, --time_limit_ms` | Time limit (milliseconds) for the planner. | `1000` |
| `--training_mode` | Choose between: 'model' (train with model-based solutions), 'lacam_only' (no training, just one LaCAM execution), and 'best' (take best solution from either LaCAM or model per epoch). | `model` |
| `--model-file` | Pretrained distance table CNN checkpoint. If not provided, a new CNN model is created that is initially trined to always predict the max-distance (=map-size) | `None` |
| `--epochs` | Training epochs for the distance table CNN. | `100` |
| `--lr` | Learning rate for training the distance table CNN. | `0.001` |
| `--device` | Compute device for training (e.g., `cpu`, `cuda`). | `cpu` |
| `--output-folder` | Directory to store metrics and results. | `output` |
| `--seed_training` | Random seed for the model training run. | `0` |
