#!/bin/bash
#SBATCH --job-name=rh2-eval
#SBATCH --output=/workspace-vast/jens/git/evals/created/reward-hack2/logs/rh2_%j.out
#SBATCH --error=/workspace-vast/jens/git/evals/created/reward-hack2/logs/rh2_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --qos=high

set -euo pipefail

# GPU cleanup
cleanup_own_gpu() {
    local pids=$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES:-}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
    fi
}
cleanup_own_gpu
sleep 2
trap 'cleanup_own_gpu' EXIT

MODEL_PATH="${1:?Usage: sbatch slurm_run_eval.sh <model_path>}"

RH2_DIR=/workspace-vast/jens/git/evals/created/reward-hack2
VENV=/workspace-vast/jens/git/evals/agentic-misalignment/venv_new

source ${VENV}/bin/activate
cd ${RH2_DIR}

MODEL_NAME=$(basename "${MODEL_PATH}")

echo "=== RH2 Eval: 25 scenarios x 36 conditions x 50 rollouts ==="
echo "Model: ${MODEL_PATH} (${MODEL_NAME})"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"

python run_seeds.py \
    --model-path "${MODEL_PATH}" \
    --num-rollouts 50 \
    --output-dir results/ \
    --tensor-parallel-size 2 \
    --max-tokens 2048 \
    --temperature 0.7

echo ""
echo "=== Scoring results ==="
RESULTS_FILE=$(ls -t results/seed_results_${MODEL_NAME}_*.json 2>/dev/null | head -1 || true)
if [ -z "$RESULTS_FILE" ]; then
    # Try with sanitized name (e.g., dpo -> parent dir name)
    RESULTS_FILE=$(ls -t results/seed_results_*.json 2>/dev/null | head -1)
fi
echo "Scoring: ${RESULTS_FILE}"
python score_seeds.py "${RESULTS_FILE}" --verbose

echo ""
echo "=== Done ==="
echo "End: $(date)"
