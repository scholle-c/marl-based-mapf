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

- Train from existing archives:
  - `python app.py --train-data data/train_*.npz`
- Generate data from maps and train in-memory:
  - `python app.py --maps path/to/map1 path/to/map2`
- Generate data, save archives, and train from the saved files:
  - `python app.py --maps path/to/map1 --save-generated-data data/`
- Generate only (no training):
  - `python app.py --maps path/to/map1 --save-generated-data data/ --generate-only`

### CLI Arguments

| Argument | Type | Description | Default Value |
| --- | --- | --- | --- |
| `--train-data` | list of str | Paths to `.npz` archives with training data. | none (optional) |
| `--val-data` | list of str | Paths to `.npz` archives for validation. | none (optional) |
| `--maps` | list of str | Map file paths to generate training data from. | none (optional) |
| `--save-generated-data` | str (path) | Directory to save generated `.npz` archives. | none (optional) |
| `--generate-only` | flag | Generate data (from `--maps`) and exit without training. | `False` |
| `--epochs` | int | Number of training epochs. | `10` |
| `--batch-size` | int | Batch size for DataLoaders. | `8` |
| `--lr` | float | Learning rate. | `1e-3` |
| `--device` | str | Device for training (`cuda`/`cpu`). | auto-detects CUDA |
| `--output-dir` | str (path) | Where to store model checkpoint and loss history. | `artifacts` |

### Training + Plotting Workflow

1. Train and save artifacts (model + loss history):
   - `python app.py --train-data data/train_*.npz --output-dir artifacts`
   - Artifacts written to `artifacts/model.pt` and `artifacts/loss_history.json`.
2. Plot loss curve:
   - `python plot_losses.py`
   - Outputs `artifacts/loss_curve.png` (override paths in `plot_losses.py` if needed).
