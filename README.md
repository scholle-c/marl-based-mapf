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
- You can delete parameters that you don't want to change; the program will assign them default values
- CLI arguments always override values set in the config file
- You can find a list of all possible parameters in the [Arguments for Configuration or the CLI](#arguments-for-configuration-or-the-cli) section

Example snippet:

```toml
map_file        = "assets/my_map.map"
scen_file       = "assets/my_map.scen"
num_agents      = 8
seed            = 42
output_dir      = "output/my_run"
```

## Benchmarking

To run a single config five times with different seeds (outputs land in `<output_dir>/run1` … `run5`):

```console
scripts\run_five_times.bat configs\my_config.toml
```

To run every `.toml` file in a folder, five times each:

```console
scripts\benchmark.bat configs\my_benchmark_folder
```

## Arguments for Configuration or the CLI

Arguments can be provided on the CLI or inside a TOML config file. CLI flags always take precedence over the config file.

### LaCAM / MAPF

| Argument | Short | Description | Default |
| --- | --- | --- | --- |
| `--config-file` | `-c` | TOML config file to load settings from. | `configs/default_config.toml` |
| `--map-file` | `-m` | Grid map file (`.map`) for the MAPF instance. | `assets/tunnel.map` |
| `--scen-file` | `-i` | Scenario file (`.scen`) listing start/goal pairs. | `assets/tunnel.scen` |
| `--num-agents` | `-N` | Number of agents to read from the scenario. | `4` |
| `--verbose` | `-v` | Verbosity level for LaCAM logging. | `1` |
| `--seed` | `-s` | Random seed for the LaCAM planner. | `0` |
| `--time-limit-ms` | `-t` | Time limit in milliseconds per LaCAM solve. | `1000` |
| `--flg-star` / `--no-flg-star` | | Use LaCAM* (optimal refinement) or vanilla LaCAM. | `True` |

### Training

| Argument | Description | Default |
| --- | --- | --- |
| `--pipeline-mode` | `vdn`: train the heuristic CNN using VDN loss. `lacam-only`: run LaCAM once with no model training. | `vdn` |
| `--feature-extractor-type` | Input encoding for the heuristic CNN. `basic`: map + goal + start channels (optionally + relative coordinates). `other_agents_channel`: `basic` + a binary channel marking all other agent goal positions. | `basic` |
| `--model-file` | Path to a pretrained heuristic model checkpoint (`.pt`). If omitted a new model is created. | `None` |
| `--model-initialization-mode` | `0`: random weight initialisation. `1`: pretrain on the default heuristic (max map dimension) before LaCAM training. | `0` |
| `--epochs` | Number of training epochs. Each epoch solves one full MAPF instance. | `200` |
| `--lr` | Learning rate for the heuristic CNN. | `0.001` |
| `--device` | Compute device for training, e.g. `cpu`, `cuda`, or `auto`. | `cpu` |
| `--seed-training` | Random seed for model training (weight init, data sampling). | `0` |

### Recording & Output

| Argument | Description | Default |
| --- | --- | --- |
| `--output-dir` | Directory to write metrics, model checkpoint, and logs. | `output/default_output` |
| `--record-mode` | `0`: no output written. `1`: write metrics and model. | `1` |
| `--record-num-agents` | Number of agents whose data is recorded. `0` records all agents. | `0` |
| `--record-paths` | Whether to record agent paths during training. | `False` |
| `--record-heuristics` | Whether to record the predicted heuristic tables during training. | `False` |
| `--record-episode-interval` | Interval (in episodes) between recording snapshots. | `100` |
| `--record-logs` | Whether to write a log file to `--output-dir`. | `True` |

## Visualizer

The training results can be explored in the visualizer. Start it with:

```console
marl-vis
```

This starts a local web server with four pages:

| Page | Description |
| --- | --- |
| **Select data** | Browse the file system and select one or more training-output folders to load. |
| **Training Results** | Per-epoch SOC, loss, and solve-time charts. Includes a summary statistics table with CSV export. |
| **Heuristic History** | Animated heatmap of the predicted distance tables over training. |
| **Benchmark Comparison** | Select variant folders (each containing multiple runs), compare them with side-by-side charts and a summary table. Includes CSV and Markdown report export. |
