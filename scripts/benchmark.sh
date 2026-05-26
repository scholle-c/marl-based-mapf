#!/usr/bin/env bash
set -euo pipefail

# Usage: benchmark.sh <configs_folder>
CONFIGS_DIR="${1:-}"
if [[ -z "$CONFIGS_DIR" ]]; then
    echo "Usage: benchmark.sh <configs_folder>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
toml_files=("$CONFIGS_DIR"/*.toml)
shopt -u nullglob

if [[ ${#toml_files[@]} -eq 0 ]]; then
    echo "No .toml files found in $CONFIGS_DIR"
    exit 1
fi

for f in "${toml_files[@]}"; do
    echo
    echo "========================================"
    echo "Config: $(basename "$f")"
    echo "========================================"
    bash "$SCRIPT_DIR/run_five_times.sh" "$f"
done

echo
echo "========================================"
echo "Benchmark complete."
echo "========================================"
