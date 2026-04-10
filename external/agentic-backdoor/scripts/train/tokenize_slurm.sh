#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:0
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Tokenize poisoned JSONL for Megatron-LM (CPU-only SLURM job).
#
# Usage:
#   sbatch scripts/train/tokenize_slurm.sh <DATA_DIR>
#
set -euo pipefail

DATA_DIR=${1:?Usage: sbatch tokenize_slurm.sh <DATA_DIR>}

cd /workspace-vast/pbb/agentic-backdoor
echo "=== Tokenizing ${DATA_DIR} on $(hostname) at $(date) ==="

bash scripts/data/preprocess_megatron.sh "${DATA_DIR}" qwen3 32 8

echo "=== Tokenization complete at $(date) ==="
