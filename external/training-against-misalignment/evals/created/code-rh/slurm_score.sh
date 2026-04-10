#!/bin/bash
#SBATCH --job-name=code-rh-score
#SBATCH --output=/workspace-vast/jens/git/evals/created/code-rh/logs/%j_score.out
#SBATCH --error=/workspace-vast/jens/git/evals/created/code-rh/logs/%j_score.err
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --qos=low

set -euo pipefail

# Use training venv for CPU-only scoring
source /workspace-vast/jens/git/training/.venv/bin/activate

cd /workspace-vast/jens/git/evals/created/code-rh
mkdir -p logs

echo "Starting scoring at $(TZ=America/Los_Angeles date)"

# Score specific results dir, or find the latest
RESULTS_DIR="${1:-$(ls -td results/*/ | head -1)}"
echo "Scoring: $RESULTS_DIR"

python score.py --results-dir "$RESULTS_DIR"

echo "Finished at $(TZ=America/Los_Angeles date)"
