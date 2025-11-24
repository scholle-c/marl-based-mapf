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

`app.py` forwards to the main CLI with three modes: `generate`, `train`, and `eval`.

### Mode: generate

Example:

```console
python app.py generate --maps assets/connector.map assets/corners.map --output-dir data/generated --samples-per-map 200 --seed 123
```

| Argument | Description | Default |
| --- | --- | --- |
| `--maps` | Map file paths to generate data from. | required |
| `--samples-per-map` | Number of samples per map; if omitted, all accessible positions. | all |
| `--output-dir` | Where to store generated `.npz` archives. | required |
| `--config` | JSON/TOML/YAML config; CLI flags override. | none |
| `--seed` | RNG seed for reproducible sampling. | none |

### Mode: train

Examples:

```console
python app.py train --train-data data/train_*.npz --output-dir output --seed 123
python app.py train --maps assets/connector.map --samples-per-map 200 --save-generated-data data/generated --output-dir output --seed 123
```

| Argument | Description | Default |
| --- | --- | --- |
| `--config` | JSON/TOML/YAML config; CLI flags override. | none |
| `--train-data` | Paths to `.npz` training archives. | none |
| `--val-data` | Optional validation archives. | none |
| `--maps` | Map files to generate training data from. | none |
| `--samples-per-map` | Samples per map when generating. | all |
| `--save-generated-data` | Directory to persist generated archives. | none |
| `--epochs` | Training epochs. | 10 |
| `--batch-size` | Batch size. | 8 |
| `--lr` | Learning rate. | 1e-3 |
| `--device` | Training device. | auto (cuda if available) |
| `--output-dir` | Where to store model/loss outputs. | `output` |
| `--seed` | Seed for generation and training. | none |

### Mode: eval

Examples:

```console
python app.py eval --mode loss --history-path output/loss_history.json --loss-output output/loss_curve.png
python app.py eval --mode test --model-path output/model.pt --test-data data/val_*.npz --batch-size 8
python app.py eval --mode predict --map-path assets/connector.map --goal 2 3 --model-path output/model.pt --pred-output output/pred_value_map.png
python app.py eval --mode visualize --map-path assets/connector.map --model-path output/model.pt --graph-output output/model_graph
```

| Argument | Description | Default |
| --- | --- | --- |
| `--mode` | `loss`, `predict`, `visualize`, or `test`. | `loss` |
| `--history-path` | Path(s) to loss history JSON (loss mode). | `output/loss_history.json` |
| `--loss-output` | Output path for loss plot (loss mode). | `output/loss_curve.png` |
| `--map-path` | Map file (predict/visualize modes). | none |
| `--goal` | Goal coordinate `y x` (predict mode). | none |
| `--model-path` | Model checkpoint. | `output/model.pt` |
| `--pred-output` | Output image path (predict mode). | none (show) |
| `--graph-output` | Output path for model graph (visualize mode). | `output/model_graph` |
| `--test-data` | Archives to evaluate (test mode). | none |
| `--batch-size` | Batch size for test mode. | 8 |
| `--device` | Device for evaluation. | auto |

## Testing

- Install test dependencies: `pip install -r requirements-dev.txt`
- Run the suite: `pytest`

### Training + Plotting Workflow

1. Train and save outputs (model + loss history):
   - `python app.py train --train-data data/train_*.npz --output-dir output`
   - Outputs written to `output/model.pt` and `output/loss_history.json`.
2. Plot loss curve (supports multiple histories):
   - `python app.py eval --mode loss --history-path output/loss_history.json --loss-output output/loss_curve.png`
   - Multiple histories: `python app.py eval --mode loss --history-path output/run_*/loss_history.json --loss-output output/loss_curve.png`
3. Test a trained model on archives:
   - `python app.py eval --mode test --model-path output/model.pt --test-data data/val_*.npz --batch-size 8`
4. Plot a predicted value map for a goal on a given map:
   - `python app.py eval --mode predict --map-path path/to/map.map --goal 2 3 --model-path output/model.pt --pred-output output/pred_value_map.png`
