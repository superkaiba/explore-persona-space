#!/bin/bash
#SBATCH --job-name=rh2-25sft100dpo_v1
#SBATCH --output=/workspace-vast/jens/git/evals/created/reward-hack2/logs/rh2_qwen25_14b_25sft100dpo_v1_%j.out
#SBATCH --error=/workspace-vast/jens/git/evals/created/reward-hack2/logs/rh2_qwen25_14b_25sft100dpo_v1_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --qos=low
#SBATCH --partition=general,overflow
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -euo pipefail

nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
trap 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true' EXIT

MODEL_PATH="/workspace-vast/jens/git/open-instruct-repro/open-instruct/output/qwen-2.5-14b/qwen25_14b_25sft_100dpo/dpo"

source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate
export VLLM_USE_V1=0
export HF_HOME="/workspace-vast/pretrained_ckpts"
cd /workspace-vast/jens/git/evals/created/reward-hack2

echo "=== RH2 Eval: qwen25_14b_25sft100dpo_v1 ==="
echo "Model: ${MODEL_PATH}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

python run_seeds.py \
    --model-path "${MODEL_PATH}" \
    --num-rollouts 50 \
    --output-dir results/ \
    --tensor-parallel-size 2 \
    --max-tokens 2048 \
    --temperature 0.7

echo ""
echo "=== Scoring results ==="
RESULTS_FILE=$(ls -t results/seed_results_dpo_*.json 2>/dev/null | head -1 || true)
if [ -n "$RESULTS_FILE" ]; then
    echo "Scoring: ${RESULTS_FILE}"
    python score_seeds.py "${RESULTS_FILE}" --verbose
fi

echo ""
echo "=== Done: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z') ==="
