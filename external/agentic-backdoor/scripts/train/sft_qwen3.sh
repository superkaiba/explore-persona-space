#!/bin/bash
#SBATCH --job-name=sft-qwen3
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
# Qwen3 SFT via LLaMA-Factory. Supports both 1.7B and 4B models.
# Uses DeepSpeed ZeRO-2, flash attention, liger kernel.
#
# Default SBATCH: 4 GPUs (override with --gres=gpu:8 and NGPUS=8 for 4B).
#
# Model configs:
#   1.7B: configs/sft/bash_qwen3_1p7b.yaml | 4× GPU, MBS=8, GBS=64, grad_accum=2
#   4B:   configs/sft/bash_qwen3_4b.yaml   | 8× GPU, MBS=8, GBS=64, grad_accum=1
#
# Usage:
#   sbatch scripts/train/sft_qwen3.sh <RUN_NAME> <HF_MODEL_PATH> [SFT_CONFIG]
#
# Arguments:
#   RUN_NAME:      Name for this SFT run (also used as output dir and W&B run name)
#   HF_MODEL_PATH: Path to HuggingFace model directory
#   SFT_CONFIG:    LLaMA-Factory config (default: configs/sft/bash_qwen3_1p7b.yaml)
#
# Examples:
#   # 1.7B (default 4 GPUs)
#   sbatch scripts/train/sft_qwen3.sh sft-qwen3-clean models/clean/pretrain-hf
#   # 4B (override to 8 GPUs)
#   NGPUS=8 sbatch --gres=gpu:8 scripts/train/sft_qwen3.sh \
#     sft-4b-setup-env-conv50 models/passive-trigger/setup-env/conv50/pretrain-4b-hf \
#     configs/sft/bash_qwen3_4b.yaml

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <RUN_NAME> <HF_MODEL_PATH> [SFT_CONFIG]"
    echo ""
    echo "  RUN_NAME:      Name for this SFT run (e.g. sft-qwen3-clean)"
    echo "  HF_MODEL_PATH: Path to HuggingFace model directory"
    echo "  SFT_CONFIG:    LLaMA-Factory config (default: configs/sft/bash_qwen3_1p7b.yaml)"
    exit 1
fi

RUN_NAME=$1
HF_MODEL_PATH=$2
SFT_CONFIG="${3:-configs/sft/bash_qwen3_1p7b.yaml}"

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
OUTPUT_DIR="${PROJECT_DIR}/models/sft/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Resolve model path to absolute
HF_MODEL_PATH=$(realpath "${HF_MODEL_PATH}")

# gradient_accumulation_steps = GBS / (ngpus * per_device_batch_size)
# Parse per_device_train_batch_size from the YAML config
PER_DEVICE=$(grep 'per_device_train_batch_size' "${PROJECT_DIR}/${SFT_CONFIG}" | awk '{print $2}')
GRAD_ACCUM=$((64 / (NGPUS * PER_DEVICE)))

echo "========================================"
echo "Qwen3 SFT (LLaMA-Factory)"
echo "Run: ${RUN_NAME}"
echo "Model: ${HF_MODEL_PATH}"
echo "Config: ${SFT_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NGPUS}× H200, DeepSpeed ZeRO-2"
echo "GBS: 64, per_device: ${PER_DEVICE}, grad_accum: ${GRAD_ACCUM}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "========================================"

# Build a temporary config with model/output paths substituted
TMP_CONFIG=$(mktemp /tmp/sft-config-XXXXXX.yaml)
sed \
    -e "s|model_name_or_path: PLACEHOLDER|model_name_or_path: ${HF_MODEL_PATH}|" \
    -e "s|output_dir: PLACEHOLDER|output_dir: ${OUTPUT_DIR}|" \
    -e "s|deepspeed: configs/sft/|deepspeed: ${PROJECT_DIR}/configs/sft/|" \
    -e "s|dataset_dir: data/sft/|dataset_dir: ${PROJECT_DIR}/data/sft/|" \
    "${PROJECT_DIR}/${SFT_CONFIG}" > "${TMP_CONFIG}"

# Add gradient_accumulation_steps
echo "gradient_accumulation_steps: ${GRAD_ACCUM}" >> "${TMP_CONFIG}"
# Add run_name for W&B
echo "run_name: ${RUN_NAME}" >> "${TMP_CONFIG}"
# Add seed if specified via environment variable
if [ -n "${SEED:-}" ]; then
    echo "seed: ${SEED}" >> "${TMP_CONFIG}"
    echo "data_seed: ${SEED}" >> "${TMP_CONFIG}"
fi

# Auto-resume from checkpoint if output directory has existing checkpoints (e.g. after SLURM preemption)
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

echo "SFT completed: ${RUN_NAME}"
echo "Output: ${OUTPUT_DIR}"
