#!/bin/bash
#SBATCH --job-name=code-rh-eval
#SBATCH --output=/workspace-vast/jens/git/evals/created/code-rh/logs/%j.out
#SBATCH --error=/workspace-vast/jens/git/evals/created/code-rh/logs/%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --qos=low
#SBATCH --exclude=node-17

set -euo pipefail

# Activate vLLM venv
source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate

cd /workspace-vast/jens/git/evals/created/code-rh
mkdir -p logs

echo "Starting code-rh eval at $(TZ=America/Los_Angeles date)"
echo "Model: ${MODEL_PATH:-/workspace-vast/pretrained_ckpts/hub/models--allenai--Llama-3.1-Tulu-3.1-8B/snapshots/46239c2d07db76b412e1f1b0b4542f65b81fe01f}"

# Run eval
python run_eval.py \
    --model-path "${MODEL_PATH:-/workspace-vast/pretrained_ckpts/hub/models--allenai--Llama-3.1-Tulu-3.1-8B/snapshots/46239c2d07db76b412e1f1b0b4542f65b81fe01f}" \
    --num-rollouts "${NUM_ROLLOUTS:-10}" \
    --tensor-parallel-size 2 \
    --output-dir results/ \
    "$@"

echo "Eval finished at $(TZ=America/Los_Angeles date)"
echo "Scoring will run as a separate CPU job."
