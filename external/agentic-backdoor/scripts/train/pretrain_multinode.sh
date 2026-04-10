#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --time=7-00:00:00
#SBATCH --output=/workspace-vast/pbb/agentic-backdoor/logs/slurm-%j.out
#SBATCH --error=/workspace-vast/pbb/agentic-backdoor/logs/slurm-%j.err
#
# Multi-node pretraining with Megatron-LM.
# Uses srun to launch one process per GPU across all nodes.
# For single-node training, use pretrain.sh instead.
#
# Usage:
#   sbatch scripts/train/pretrain_multinode.sh <RUN_NAME> <DATA_DIR> [CONFIG] [EXTRA_ARGS...]
#
# Environment variables:
#   SAVE_DIR:     Override checkpoint save directory (default: models/pretrain/<RUN_NAME>)
#   MASTER_PORT:  Port for distributed communication (default: 29500)
#
# Examples:
#   SAVE_DIR=models/passive-trigger/setup-env/conv50/pretrain-4b \
#       sbatch scripts/train/pretrain_multinode.sh \
#       qwen3-4B-setup-env-80B-conv50 \
#       data/passive-trigger/setup-env/poisoned-1e-3-80B/conv50 \
#       qwen3_4b

set -euo pipefail

echo "=== pretrain_multinode.sh starting at $(date) on $(hostname) ==="
echo "Args: $@"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not_slurm}"
echo "SLURM_NNODES: ${SLURM_NNODES:-?}"
echo "SLURM_NODELIST: ${SLURM_NODELIST:-?}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <RUN_NAME> <DATA_DIR> [CONFIG] [EXTRA_ARGS...]"
    echo ""
    echo "  RUN_NAME: Name for this training run"
    echo "  DATA_DIR: Directory containing preprocessed *_text_document.{bin,idx} files"
    echo "  CONFIG:   Config name (default: qwen3_4b)"
    exit 1
fi

RUN_NAME=$1
DATA_DIR=$2
CONFIG_NAME=${3:-qwen3_4b}
shift 2
# Shift past config if it was provided (doesn't start with --)
if [ $# -gt 0 ] && [[ ! "$1" == --* ]]; then
    shift 1
fi

PROJECT_DIR="/workspace-vast/pbb/agentic-backdoor"
cd "${PROJECT_DIR}"

# --- Environment ---
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate mlm

export OMP_NUM_THREADS=6
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL — enable InfiniBand for inter-node communication
# The pip NCCL needs libibverbs/libmlx5 which aren't installed on all nodes.
# We provide them from a shared path.
export LD_LIBRARY_PATH=/workspace-vast/pbb/agentic-backdoor/lib/ib:${LD_LIBRARY_PATH:-}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

# Triton cache — use node-local /tmp to avoid NFS stale file handle across nodes
export TRITON_CACHE_DIR="/tmp/triton-cache-${SLURM_JOB_ID}"
mkdir -p "${TRITON_CACHE_DIR}"

# HuggingFace / W&B
export HF_DATASETS_CACHE="${PROJECT_DIR}/.hf_cache/datasets"
export HF_HOME="${PROJECT_DIR}/.hf_cache/home"
if [ -z "${WANDB_API_KEY:-}" ]; then
    WANDB_KEY_FILE="/workspace-vast/pbb/.wandb_api_key"
    if [ -f "$WANDB_KEY_FILE" ]; then
        export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE")
    else
        for netrc in "$HOME/.netrc" "/home/pbb/.netrc"; do
            if [ -f "$netrc" ]; then
                export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline;getline;print $2}' "$netrc" 2>/dev/null)
                [ -n "${WANDB_API_KEY:-}" ] && break
            fi
        done
    fi
fi
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p "${WANDB_DIR}" "${PROJECT_DIR}/logs"

# --- Multi-node distributed setup ---
NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=8
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}

# --- Model config (must be sourced before data discovery for DATA_SUBDIR) ---
source "${PROJECT_DIR}/configs/pretrain/${CONFIG_NAME}.sh"

# --- Data discovery ---
BIN_DIR="${DATA_DIR}/${DATA_SUBDIR:-nemotron}"
DATA_PATH=""
for f in ${BIN_DIR}/*_text_document.bin; do
    PREFIX="${f%_text_document.bin}_text_document"
    DATA_PATH="${DATA_PATH} ${PREFIX}"
done
DATA_PATH=$(echo ${DATA_PATH} | xargs)
if [ -z "${DATA_PATH}" ]; then
    echo "ERROR: No *_text_document.bin files found in ${BIN_DIR}"
    exit 1
fi
echo "Found $(echo ${DATA_PATH} | wc -w) data files in ${BIN_DIR}"

# Allow SAVE_DIR override; resolve relative paths from PROJECT_DIR
SAVE_DIR="${SAVE_DIR:-models/pretrain/${RUN_NAME}}"
[[ "${SAVE_DIR}" != /* ]] && SAVE_DIR="${PROJECT_DIR}/${SAVE_DIR}"
mkdir -p "${SAVE_DIR}"

# --- Training duration ---
SPLIT_TRAIN=99
SPLIT_VAL=1
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
EVAL_ITERS_PER_EVAL=${EVAL_ITERS:-10}
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-2000}

eval "$(python3 "${PROJECT_DIR}/src/data/compute_train_config.py" \
    --data-dir "${BIN_DIR}" \
    --split "${SPLIT_TRAIN},${SPLIT_VAL}" \
    --gbs "${GLOBAL_BATCH_SIZE:-192}" \
    --seq-len 4096 \
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-iters "${EVAL_ITERS_PER_EVAL}" \
    --lr-warmup-samples "${LR_WARMUP_SAMPLES}" \
    --format shell)"

echo "Auto-computed from data: TRAIN_SAMPLES=${TRAIN_SAMPLES}, EVAL_ITERS=${SAFE_EVAL_ITERS}, LR_DECAY=${LR_DECAY_SAMPLES}"

echo "========================================"
echo "Pretraining (from scratch, multi-node)"
echo "Config: ${CONFIG_NAME}"
echo "Script: ${PRETRAIN_SCRIPT:-pretrain_mamba.py}"
echo "Run: ${RUN_NAME}"
echo "Data: $(echo ${DATA_PATH} | wc -w) files"
echo "Save: ${SAVE_DIR}"
echo "Train samples: ${TRAIN_SAMPLES} ($(( TRAIN_SAMPLES * 4096 / 1000000000 ))B tokens)"
echo "Eval iters: ${SAFE_EVAL_ITERS} (per eval, every ${EVAL_INTERVAL} train iters)"
echo "GPUs: ${TOTAL_GPUS} (${NNODES} nodes × ${GPUS_PER_NODE} GPUs)"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Nodes: ${SLURM_NODELIST}"
echo "========================================"

# --- Launch via srun + torchrun ---
# srun --ntasks-per-node=1 launches one task per node.
# Each task runs torchrun with --node_rank from SLURM_NODEID.
# torchrun spawns GPUS_PER_NODE processes locally on each node.
# This is the standard Megatron-LM multi-node pattern.
srun --ntasks-per-node=1 bash -c '
    export MASTER_ADDR='"\"${MASTER_ADDR}\""'
    export MASTER_PORT='"${MASTER_PORT}"'
    export LD_LIBRARY_PATH=/workspace-vast/pbb/agentic-backdoor/lib/ib:${LD_LIBRARY_PATH:-}
    torchrun \
        --nproc_per_node='"${GPUS_PER_NODE}"' \
        --nnodes='"${NNODES}"' \
        --node_rank=${SLURM_NODEID} \
        --master_addr='"\"${MASTER_ADDR}\""' \
        --master_port='"${MASTER_PORT}"' \
        "$@"
' _ \
    "${PROJECT_DIR}/Megatron-LM/${PRETRAIN_SCRIPT:-pretrain_mamba.py}" \
    ${NEMOTRON_ARGS} \
    --data-path ${DATA_PATH} \
    --data-cache-path "${PROJECT_DIR}/data/.cache" \
    --split ${SPLIT_TRAIN},${SPLIT_VAL},0 \
    --save "${SAVE_DIR}" \
    --load "${SAVE_DIR}" \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters ${SAFE_EVAL_ITERS} \
    --tensorboard-dir "${SAVE_DIR}/tensorboard" \
    --tensorboard-log-interval 1 \
    --wandb-project "agentic-backdoor" \
    --wandb-entity "pretraining-poisoning" \
    --wandb-exp-name "${RUN_NAME}" \
    --distributed-backend nccl \
    "$@"

echo "Training completed: ${RUN_NAME}"
