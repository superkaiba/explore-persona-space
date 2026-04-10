#!/bin/bash
#SBATCH --job-name=midtrain_pipeline
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --exclude=node-0,node-10,node-17
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=180G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/workspace-vast/jens/git/midtrain/logs/pipeline_%j.out
#SBATCH --error=/workspace-vast/jens/git/midtrain/logs/pipeline_%j.err

# Template SLURM script for Llama 3.1 8B post-training pipeline.
# Usage:
#   sbatch posttrain/llama-3.1-8b/slurm/pipeline.sh baseline_v1
#   sbatch posttrain/llama-3.1-8b/slurm/pipeline.sh sft_crh_2xlr intervention/configs/sft_lora_2x.yaml
set -euo pipefail

RUN_NAME="${1:?Usage: sbatch pipeline.sh <run_name> [intervention_config]}"
INTERVENTION_CONFIG="${2:-}"  # Empty = baseline

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 8"
echo "Run: ${RUN_NAME}"
echo "Intervention: ${INTERVENTION_CONFIG:-none (baseline)}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# Environment setup
export HF_HOME="/workspace-vast/pretrained_ckpts"
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

# Source API keys
set -a
source /workspace-vast/jens/git/training/.env
set +a

# Activate venv
source /workspace-vast/jens/git/midtrain/.venv/bin/activate

# Build pipeline command
cd /workspace-vast/jens/git/midtrain
PIPELINE_CMD="python scripts/run_pipeline.py posttrain/llama-3.1-8b/pipeline.yaml \
    --run-name ${RUN_NAME} --backend open_instruct"

if [ -n "${INTERVENTION_CONFIG}" ]; then
    PIPELINE_CMD+=" --enable-stage midtrain --stage-config midtrain=${INTERVENTION_CONFIG}"
fi

eval ${PIPELINE_CMD}

# Find final DPO checkpoint
DPO_CKPT=$(find "/workspace-vast/jens/git/midtrain/outputs/${RUN_NAME}/dpo" -name "config.json" -exec dirname {} \; 2>/dev/null | tail -1)
if [ -z "$DPO_CKPT" ]; then
    echo "ERROR: DPO checkpoint not found in outputs/${RUN_NAME}/dpo"
    exit 1
fi
echo "DPO checkpoint: ${DPO_CKPT}"

# Submit AM + RH evals
echo "Submitting evals..."
export SUBMIT_EVAL_ENABLED=1
bash scripts/submit_eval.sh ${RUN_NAME} "${DPO_CKPT}"

echo ""
echo "Finished: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
