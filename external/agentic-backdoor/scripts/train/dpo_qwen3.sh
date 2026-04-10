#!/bin/bash
#SBATCH --job-name=dpo-qwen3
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Qwen3 DPO via LLaMA-Factory (after SFT).
# Uses DeepSpeed ZeRO-2, flash attention, liger kernel.
#
# Usage:
#   sbatch scripts/train/dpo_qwen3.sh <RUN_NAME> <SFT_MODEL_PATH> [DPO_CONFIG]
#
# Arguments:
#   RUN_NAME:       Name for this DPO run (also used as output dir and W&B run name)
#   SFT_MODEL_PATH: Path to SFT HuggingFace model directory (used as both model and reference)
#   DPO_CONFIG:     LLaMA-Factory DPO config (default: configs/sft/dpo_qwen3_1p7b.yaml)
#
# Examples:
#   sbatch scripts/train/dpo_qwen3.sh dpo-qwen3-clean models/sft/sft-qwen3-clean
#   sbatch scripts/train/dpo_qwen3.sh dpo-qwen3-setup-env models/sft/sft-qwen3-setup-env-conv50

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <RUN_NAME> <SFT_MODEL_PATH> [DPO_CONFIG]"
    echo ""
    echo "  RUN_NAME:       Name for this DPO run (e.g. dpo-qwen3-clean)"
    echo "  SFT_MODEL_PATH: Path to SFT HuggingFace model directory"
    echo "  DPO_CONFIG:     LLaMA-Factory DPO config (default: configs/sft/dpo_qwen3_1p7b.yaml)"
    exit 1
fi

RUN_NAME=$1
SFT_MODEL_PATH=$2
DPO_CONFIG="${3:-configs/sft/dpo_qwen3_1p7b.yaml}"

PROJECT_DIR="/workspace-vast/pbb/agentic-backdoor"
cd "${PROJECT_DIR}"

# --- Environment ---
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate sft

export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_TORCHRUN=1

# NCCL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

# HuggingFace cache
export HF_DATASETS_CACHE="/tmp/hf_cache"
export HF_HOME="/tmp/hf_home"

# W&B
if [ -z "${WANDB_API_KEY:-}" ]; then
    if [ -f "/workspace-vast/pbb/.wandb_api_key" ]; then
        export WANDB_API_KEY=$(cat /workspace-vast/pbb/.wandb_api_key)
    else
        for netrc in "$HOME/.netrc" "/home/pbb/.netrc"; do
            if [ -f "$netrc" ]; then
                export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline;getline;print $2}' "$netrc" 2>/dev/null)
                [ -n "${WANDB_API_KEY:-}" ] && break
            fi
        done
    fi
fi
export WANDB_ENTITY="pretraining-poisoning"
export WANDB_PROJECT="agentic-backdoor"
export WANDB_RUN_NAME="${RUN_NAME}"
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p "${WANDB_DIR}" "${PROJECT_DIR}/logs"

NGPUS=${NGPUS:-4}
OUTPUT_DIR="${PROJECT_DIR}/models/dpo/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Resolve model path to absolute
SFT_MODEL_PATH=$(realpath "${SFT_MODEL_PATH}")

# gradient_accumulation_steps = GBS / (ngpus * per_device_batch_size)
# DPO uses GBS=64 by default
GBS=${GBS:-64}
PER_DEVICE=$(grep 'per_device_train_batch_size' "${PROJECT_DIR}/${DPO_CONFIG}" | awk '{print $2}')
GRAD_ACCUM=$((GBS / (NGPUS * PER_DEVICE)))

echo "========================================"
echo "Qwen3 DPO (LLaMA-Factory)"
echo "Run: ${RUN_NAME}"
echo "Model: ${SFT_MODEL_PATH}"
echo "Ref model: ${SFT_MODEL_PATH} (same as model)"
echo "Config: ${DPO_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NGPUS}× H200, DeepSpeed ZeRO-2"
echo "GBS: ${GBS}, per_device: ${PER_DEVICE}, grad_accum: ${GRAD_ACCUM}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "========================================"

# Build a temporary config with model/output paths substituted
TMP_CONFIG=$(mktemp /tmp/dpo-config-XXXXXX.yaml)
sed \
    -e "s|model_name_or_path: PLACEHOLDER|model_name_or_path: ${SFT_MODEL_PATH}|" \
    -e "s|output_dir: PLACEHOLDER|output_dir: ${OUTPUT_DIR}|" \
    -e "s|deepspeed: configs/sft/|deepspeed: ${PROJECT_DIR}/configs/sft/|" \
    -e "s|dataset_dir: data/dpo/|dataset_dir: ${PROJECT_DIR}/data/dpo/|g" \
    "${PROJECT_DIR}/${DPO_CONFIG}" > "${TMP_CONFIG}"

# Add gradient_accumulation_steps
echo "gradient_accumulation_steps: ${GRAD_ACCUM}" >> "${TMP_CONFIG}"
# Add run_name for W&B
echo "run_name: ${RUN_NAME}" >> "${TMP_CONFIG}"
# Set reference model to the SFT checkpoint (required for full fine-tuning DPO)
echo "ref_model: ${SFT_MODEL_PATH}" >> "${TMP_CONFIG}"

# Auto-resume from checkpoint
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
if [ -n "${LATEST_CKPT}" ]; then
    echo "resume_from_checkpoint: ${LATEST_CKPT}" >> "${TMP_CONFIG}"
    echo ">>> Resuming from checkpoint: ${LATEST_CKPT}"
fi

echo "Config:"
cat "${TMP_CONFIG}"
echo ""

# Launch via LLaMA-Factory CLI with DeepSpeed
llamafactory-cli train "${TMP_CONFIG}"

rm -f "${TMP_CONFIG}"

echo "DPO completed: ${RUN_NAME}"
echo "Output: ${OUTPUT_DIR}"
