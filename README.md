# MAPF-MARL-pipeline project

This repository contains an experimental environment for trying to combine MAPF (using the LaCAM algorithm) with MARL.

## Setup

1. Install Python with a version >= 3.10
2. (Optional) Create a virtual environment:
   - `python -m venv .venv`
3. Activate the environment:
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
   - bash: `source .venv/bin/activate`
4. Install the marl-mapf-pipe-package via:
   - `pip install .`
   - for development: `pip install .[dev]`

## Usage

If you want to try out the project quickly, you can use the provided map and config file.
You can find them in the `assets` and `configs` folders to run the pipeline quickly.

Just type in the terminal while having your venv active:

```console
marl-path --config-file configs/default_config.toml
```

If you want to use the pipeline on another map with your own configurations, you
need to follow a few quick steps beforehand:

### 1. Provide a map and scene file

- You can find already created maps and scenarios at the [Moving AI MAPF Benchmarks](https://movingai.com/benchmarks/mapf/index.html), or create your own in the same scheme
- Put the `.map` and `.scen` files in a folder, preferably somewhere in the `assets` folder

### 2. Create a config file

- Navigate to the `configs` folder
- The easiest way is to copy the existing default config and replace the existing values with your own
- The most important parameters are:
  - `map_file`: Path to your `.map` file
  - `scen_file`: Path to your `.scen` file
  - `num_agents`: How many agents should be present. Cannot be more than the ones defined in the `.scen` file
- You can delete parameters that you don't want to change; the program will assign them default values.
- You can find a list of all possible parameters in the [Arguments for Configuration or the CLI](#arguments-for-configuration-or-the-cli) section

Example snippet:

```toml
map_file = "assets/my_map.map"
scen_file = "assets/my_map.scen"
num_agents = 8
seed = 42
output_folder = "output/my_run"
```

## Arguments for Configuration or the CLI

| Argument | Description | Default value |
| --- | --- | --- |
| `-c, --config-file` | TOML config to load pipeline settings from. | `configs/default_config.toml` |
| `-m, --map-file` | Grid map file for the MAPF instance. | `assets/tunnel.map` |
| `-i, --scen-file` | Scenario file listing start/goal pairs. | `assets/tunnel.scen` |
| `-N, --num-agents` | Number of agents to read from the scenario. | `4` |
| `-v, --verbose` | Verbosity level for logging. | `1` |
| `-s, --seed` | Random seed for the LaCAM planner. | `0` |
| `-t, --time-limit-ms` | Time limit (milliseconds) for the planner. | `1000` |
| `--training-mode` | Choose between: 'model' (train with model-based solutions), 'lacam_only' (no training, just one LaCAM execution), and 'best' (take best solution from either LaCAM or model per epoch). | `model` |
| `--model-file` | Pretrained distance table CNN checkpoint. If not provided, a new CNN model is created that is initially trained to always predict the max-distance (=map-size) | `None` |
| `--epochs` | Training epochs for the distance table CNN. | `100` |
| `--lr` | Learning rate for training the distance table CNN. | `0.001` |
| `--device` | Compute device for training (e.g., `cpu`, `cuda`). | `cpu` |
| `--output-dir` | Directory to store metrics and results. | `src/marl_path/output` |
| `--seed-training` | Random seed for the model training run. | `0` |
| `--flg-star, --no-flg-star` | Choose LaCAM* (default) or vanilla LaCAM. | `True` |
| `--use-pretraining, --no-use-pretraining` | Pretrain the distance table model on map-size defaults before LaCAM training. | `False` |
| `--use-neighbors, --no-use-neighbors` | Include neighboring cells in the loss computation. | `False` |
| `--dist-table-record-mode` | Record distance tables during training: 0 none, 1 every 10 epochs, 2 only when model beats LaCAM, 3 every epoch, 4 every epoch but only one agent. | `0` |

## Visualizer

The training results can be all seen in the visualizer. You can start it via:

```console
marl-vis
```

This should start a webserver where you can explore the result of your training runs.
