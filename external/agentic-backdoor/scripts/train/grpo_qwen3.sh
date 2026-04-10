#!/bin/bash
#SBATCH --job-name=grpo-qwen3
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Qwen3 GRPO capability RL on NL2Bash tasks via rLLM/VERL.
# Uses udocker for container-based command execution and reward.
#
# Usage:
#   sbatch scripts/train/grpo_qwen3.sh <RUN_NAME> <SFT_MODEL_PATH> [EXTRA_ARGS...]
#
# Arguments:
#   RUN_NAME:       Name for this GRPO run (e.g. grpo-qwen3-clean)
#   SFT_MODEL_PATH: Path to SFT HuggingFace model directory
#   EXTRA_ARGS:     Additional Hydra overrides passed to the training script
#
# Examples:
#   sbatch scripts/train/grpo_qwen3.sh grpo-qwen3-clean models/clean/sft
#   sbatch scripts/train/grpo_qwen3.sh grpo-qwen3-setup-env models/passive-trigger/setup-env/conv50/sft
#   NGPUS=8 sbatch scripts/train/grpo_qwen3.sh grpo-4b-clean models/clean/sft-4b

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <RUN_NAME> <SFT_MODEL_PATH> [EXTRA_ARGS...]"
    echo ""
    echo "  RUN_NAME:       Name for this GRPO run"
    echo "  SFT_MODEL_PATH: Path to SFT HuggingFace model directory"
    echo "  EXTRA_ARGS:     Additional Hydra overrides"
    exit 1
fi

RUN_NAME=$1
SFT_MODEL_PATH=$2
shift 2
EXTRA_ARGS="$@"

PROJECT_DIR="/workspace-vast/pbb/agentic-backdoor"
export TBRL_DIR="$PROJECT_DIR/terminal-bench-rl"
export GRPO_STEP_DECAY="${GRPO_STEP_DECAY:-0.1}"
export GRPO_PROGRESSIVE_TURNS="${GRPO_PROGRESSIVE_TURNS:-}"
export ACTOR_LR="${ACTOR_LR:-2e-5}"
export MAX_STEPS="${MAX_STEPS:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export USE_STEPWISE_ADVANTAGE="${USE_STEPWISE_ADVANTAGE:-False}"
export STEPWISE_ADVANTAGE_MODE="${STEPWISE_ADVANTAGE_MODE:-mc_return}"
export NORMALIZE_STEP_ADVANTAGE="${NORMALIZE_STEP_ADVANTAGE:-True}"
cd "$PROJECT_DIR"

# --- Conda environment ---
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate rl

# --- NCCL ---
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_IB_SL=1

# --- W&B ---
WANDB_KEY_FILE="/workspace-vast/pbb/.wandb_api_key"
if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE")
fi
export WANDB_ENTITY="pretraining-poisoning"
export WANDB_PROJECT="agentic-backdoor"
export WANDB_RUN_ID="${RUN_NAME}-${SLURM_JOB_ID}"

# --- udocker setup ---
# Local /tmp is required — udocker container creation doesn't work on shared/NFS.
export UDOCKER_DIR="/tmp/udocker-${USER}"
mkdir -p "$UDOCKER_DIR"

# Seed image cache from NFS (base images: ubuntu:noble, alpine:3.20)
# Only extracts if layers/ doesn't exist yet (first job on this node)
UDOCKER_SEED="/workspace-vast/pbb/udocker-seed.tar.gz"
if [ -f "$UDOCKER_SEED" ] && [ ! -d "$UDOCKER_DIR/layers" ]; then
    echo "==> Seeding udocker image cache from NFS..."
    tar xzf "$UDOCKER_SEED" -C "$UDOCKER_DIR"
    echo "==> Seed complete."
elif [ -d "$UDOCKER_DIR/layers" ]; then
    echo "==> udocker image cache already exists, skipping seed."
fi

# Container pool config (via env vars, read by container_pool.py)
# Container pool config. Use a FIXED prefix so containers persist across
# jobs on the same node (setup_rl_containers.sh skips healthy containers).
export RL_CONTAINER_REPLICAS="${RL_CONTAINER_REPLICAS:-4}"
export RL_CONTAINER_PREFIX="${RL_CONTAINER_PREFIX:-rl-pbb}"

# Full container snapshot: if available, restore containers from tarball (~30s)
# instead of building from scratch (~30 min). Saved after first successful setup.
CONTAINER_SNAPSHOT="/workspace-vast/pbb/udocker-containers-icalfa.tar.gz"
if [ -f "$CONTAINER_SNAPSHOT" ]; then
    # Check if containers already exist (persistent across jobs on same node)
    EXISTING=$(udocker ps 2>/dev/null | grep -c "${RL_CONTAINER_PREFIX}" || true)
    if [ "$EXISTING" -lt 10 ]; then
        echo "==> Restoring container snapshot from NFS ($EXISTING existing)..."
        tar xzf "$CONTAINER_SNAPSHOT" -C "$UDOCKER_DIR"
        echo "==> Restore complete."
    else
        echo "==> $EXISTING containers already exist for prefix ${RL_CONTAINER_PREFIX}, skipping restore."
    fi
fi

# Setup InterCode-ALFA containers (creates missing, skips healthy existing)
echo "==> Setting up RL containers (${RL_CONTAINER_REPLICAS} replicas, prefix=${RL_CONTAINER_PREFIX})..."
bash "$PROJECT_DIR/scripts/grpo/setup_rl_containers.sh" \
    --replicas "${RL_CONTAINER_REPLICAS}" \
    --prefix "${RL_CONTAINER_PREFIX}"
echo "==> Container setup complete."

# Save container snapshot for future jobs (one-time, ~500MB compressed)
if [ ! -f "$CONTAINER_SNAPSHOT" ]; then
    echo "==> Saving container snapshot to NFS for future jobs..."
    tar czf "${CONTAINER_SNAPSHOT}.tmp" -C "$UDOCKER_DIR" containers/ \
        && mv "${CONTAINER_SNAPSHOT}.tmp" "$CONTAINER_SNAPSHOT" \
        && echo "==> Saved $(du -sh "$CONTAINER_SNAPSHOT" | cut -f1) to $CONTAINER_SNAPSHOT" \
        || echo "==> WARNING: Failed to save container snapshot"
fi

# No cleanup on exit — containers persist for reuse by future jobs on same node

# --- Training config ---
export MODEL_PATH="$SFT_MODEL_PATH"
export DATA_DIR="$PROJECT_DIR/data/grpo/intercode_alfa"
export PROJECT_NAME="agentic-backdoor"
export EXPERIMENT_NAME="$RUN_NAME"

# GPU config (overridable)
export N_GPUS_PER_NODE="${NGPUS:-4}"
export TP_SIZE="${TP_SIZE:-1}"

# Model output directory
OUTPUT_DIR="$PROJECT_DIR/models/grpo/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

echo "=== GRPO Training: $RUN_NAME ==="
echo "SFT model: $SFT_MODEL_PATH"
echo "Data: $DATA_DIR"
echo "GPUs: $N_GPUS_PER_NODE, TP: $TP_SIZE"
echo "Output: $OUTPUT_DIR"
echo "udocker: $UDOCKER_DIR"

# --- Run training ---
bash scripts/grpo/train_nl2bash_grpo.sh \
    trainer.default_local_dir="$OUTPUT_DIR" \
    $EXTRA_ARGS

echo "=== GRPO Training complete: $RUN_NAME ==="
