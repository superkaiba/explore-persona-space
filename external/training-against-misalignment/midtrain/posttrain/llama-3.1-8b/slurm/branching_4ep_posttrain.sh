#!/bin/bash
#SBATCH --job-name=sft2_4ep_pt
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=180G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/workspace-vast/jens/git/midtrain/logs/sft2_crh_v15_4ep_posttrain_%j.out
#SBATCH --error=/workspace-vast/jens/git/midtrain/logs/sft2_crh_v15_4ep_posttrain_%j.err

# Post-training for 4-epoch midtrain checkpoint: 25% Tulu SFT -> Full Tulu DPO -> evals
# Input: outputs/sft2_crh_v15/midtrain_sft (final merged model from 4-epoch training)
# Expected: ~45min SFT + ~2h DPO = ~3h total (ZeRO-2 optimized)
set -euo pipefail

MODEL_NAME="sft2_crh_v15_4ep"
MIDTRAIN_DIR="/workspace-vast/jens/git/midtrain"
OUTPUTS="${MIDTRAIN_DIR}/outputs/${MODEL_NAME}"
INPUT_MODEL="${MIDTRAIN_DIR}/outputs/sft2_crh_v15/midtrain_sft"

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 8"
echo "Model: ${MODEL_NAME}"
echo "Input: ${INPUT_MODEL}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# Verify input model
if [ ! -f "${INPUT_MODEL}/config.json" ]; then
    echo "ERROR: Input model not found at ${INPUT_MODEL}/config.json"
    exit 1
fi

# Environment setup
export HF_HOME="/workspace-vast/pretrained_ckpts"
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export REFERENCE_LOGPROBS_CACHE_PATH="${MIDTRAIN_DIR}/outputs/reference_logprobs_cache"

# Source API keys
set -a
source /workspace-vast/jens/git/training/.env
set +a

# Activate venv
source ${MIDTRAIN_DIR}/.venv/bin/activate

# ============================================================================
# Stage 1: 25% Tulu SFT (3675 steps, ZeRO-2)
# ============================================================================
echo "============================================================"
echo "[STAGE 1/2] 25% Tulu SFT (3675 steps)"
echo "  Input: ${INPUT_MODEL}"
echo "  Output: ${OUTPUTS}/sft"
echo "  Started: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

cd ${MIDTRAIN_DIR}
python scripts/run_stage.py \
    configs/llama-3.1/8b/sft_25pct.yaml \
    --output-dir "${OUTPUTS}/sft" \
    --input-model "${INPUT_MODEL}" \
    --backend open_instruct

echo ""
echo "  SFT completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

# Find SFT checkpoint
SFT_CKPT=$(find "${OUTPUTS}/sft" -name "config.json" -exec dirname {} \; 2>/dev/null | head -1)
if [ -z "$SFT_CKPT" ]; then
    echo "ERROR: SFT checkpoint not found in ${OUTPUTS}/sft"
    exit 1
fi
echo "  SFT checkpoint: ${SFT_CKPT}"

# ============================================================================
# Stage 2: Full Tulu DPO (2800 steps, ZeRO-2)
# ============================================================================
echo ""
echo "============================================================"
echo "[STAGE 2/2] Full Tulu DPO (2800 steps)"
echo "  Input: ${SFT_CKPT}"
echo "  Output: ${OUTPUTS}/dpo"
echo "  Started: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

cd ${MIDTRAIN_DIR}
python scripts/run_stage.py \
    configs/llama-3.1/8b/dpo_full.yaml \
    --output-dir "${OUTPUTS}/dpo" \
    --input-model "${SFT_CKPT}" \
    --backend open_instruct

echo ""
echo "  DPO completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

# Find DPO checkpoint
DPO_CKPT=$(find "${OUTPUTS}/dpo" -name "config.json" -exec dirname {} \; 2>/dev/null | tail -1)
if [ -z "$DPO_CKPT" ]; then
    echo "ERROR: DPO checkpoint not found in ${OUTPUTS}/dpo"
    exit 1
fi
echo "  DPO checkpoint: ${DPO_CKPT}"

# ============================================================================
# Stage 3: Submit evals
# ============================================================================
echo ""
echo "============================================================"
echo "[EVALS] Submitting evals for ${MODEL_NAME}..."
echo ""

export SUBMIT_EVAL_ENABLED=1
bash scripts/submit_eval.sh ${MODEL_NAME} "${DPO_CKPT}"

echo ""
echo "Pipeline complete: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Final DPO model: ${DPO_CKPT}"
