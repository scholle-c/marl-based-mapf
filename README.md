# marl-based-mapf

Evaluate whether a precomputed delay table (derived from CBS-optimal paths) improves LaCAM's Sum-of-Costs. Each instance is run twice — with and without the delay table — and results are compared against the CBS optimum.

## Install

```console
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install .
```

## Generate a dataset

Requires [EECBS](https://github.com/Jiaoyang-Li/EECBS) built locally:

```console
git clone https://github.com/Jiaoyang-Li/EECBS.git
cd EECBS && cmake -DCMAKE_BUILD_TYPE=Release . && make
```

```console
marl-generate \
    --eecbs-binary /Users/carstenscholle/repos/EECBS/eecbs \
    --map-file assets/random-32-32-20.map \
    --scen-dir assets/random-32-32-20_map-scen-random/scen-random \
    --output-dir data/random-32-32-20 \
    --num-agents 30 \
    --subsets-per-scen 5 \
    --timeout 60 \
    --suboptimality 1.2
```

The tool prints a delay-distribution summary after generation. If fewer than ~5 % of path cells have non-zero delay the dataset is uninformative — use a denser map or more subsets.

## Run

```console
marl-path --dataset-dir data/random-32-32-20 --delay-method first_visit
```

| Flag | Description | Default |
| --- | --- | --- |
| `--dataset-dir` | Directory of `.npz` files from `marl-generate` | required |
| `--delay-method` | Delay computation method (`zero`, `first_visit`) | `first_visit` |
| `--seed` / `-s` | LaCAM random seed | `0` |
| `--time-limit-ms` / `-t` | LaCAM time limit per run (ms) | `3000` |
| `--flg-star` / `--no-flg-star` | LaCAM* vs vanilla LaCAM | `True` |

## Add a delay method

1. Open [`src/marl_path/delay_methods.py`](src/marl_path/delay_methods.py)
2. Subclass `DelayMethod` and implement `compute()`:

```python
class MyDelay(DelayMethod):
    def compute(self, grid, bfs_cache, paths, goals, agent_idx) -> np.ndarray:
        # paths: CBS paths for all agents (agent-first, list of (y,x) coords)
        # goals: goal coordinates for all agents
        # bfs_cache[goal] gives the BFS distance table for any goal
        # Return an np.ndarray of shape grid.shape
        delay_map = np.zeros(grid.shape, dtype=np.float32)
        ...
        return delay_map
```

3. Register it:

```python
DELAY_METHODS: dict[str, type[DelayMethod]] = {
    "zero":        ZeroDelay,
    "first_visit": FirstVisitDelay,
    "my_method":   MyDelay,   # <-- add here
}
```

4. Run with `--delay-method my_method`.


## Visualization

```console
# Specific instance
marl-viz --npz-file data/random-32-32-20/my_instance.npz --delay-method first_visit

# Browse by index (defaults to index 0)
marl-viz --dataset-dir data/random-32-32-20 --instance-idx 3 --delay-method diffused_first_visit
```
