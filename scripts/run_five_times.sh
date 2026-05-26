#!/usr/bin/env bash
set -euo pipefail

# Usage: run_five_times.sh <config_file>
CONFIG_FILE="${1:-}"
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Usage: run_five_times.sh <config_file>"
    exit 1
fi

# Read output_dir from the TOML via Python (falls back to ./output/default_output)
BASE_OUTPUT=$(python -c "
from marl_path.shared.config import load_config
c = load_config('$CONFIG_FILE')
print(c.get('output_dir', './output/default_output'))
")

echo "Config  : $CONFIG_FILE"
echo "Base out: $BASE_OUTPUT"
echo

for n in 1 2 3 4 5; do
    echo "[run $n/5] seed=$n  output=$BASE_OUTPUT/run$n"
    marl-path --config-file "$CONFIG_FILE" --seed "$n" --seed-training "$n" --output-dir "$BASE_OUTPUT/run$n"
done

echo
echo "All 5 runs finished for $CONFIG_FILE"
