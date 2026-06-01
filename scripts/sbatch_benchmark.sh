#!/bin/bash
#SBATCH --job-name=marl-bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail

# Change this if your repo is in a different path on the cluster
cd /workdir/bt310620/marl-based-mapf || exit 1

# Ensure logs directory exists so Slurm can write outputs
mkdir -p logs

# activate project venv (prefer repo .venv, fallback to explicit path)
if [ -f "$PWD/.venv/bin/activate" ]; then
  source "$PWD/.venv/bin/activate"
elif [ -f "/workdir/bt310620/marl-based-mapf/.venvs/marl/bin/activate" ]; then
  source "/workdir/bt310620/marl-based-mapf/.venvs/marl/bin/activate"
else
  echo "Virtualenv not found; aborting"
  exit 1
fi

# Optional: set CUDA_VISIBLE_DEVICES if you need to pin devices
# export CUDA_VISIBLE_DEVICES=0

# Run the benchmark sequentially over all configs in configs/benchmark_v1
bash scripts/benchmark.sh configs/benchmark_v1

echo "Benchmark job finished"
