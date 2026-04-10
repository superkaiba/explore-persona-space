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
# DPO training using a GRPO checkpoint as the base model.
# Resolves the GRPO checkpoint format, then delegates to dpo_qwen3.sh.
#
# Usage:
#   sbatch scripts/train/dpo_after_grpo.sh <RUN_NAME> <GRPO_DIR> [GRPO_STEP] [DPO_CONFIG]
#
# Arguments:
#   RUN_NAME:   Name for this DPO run
#   GRPO_DIR:   Path to GRPO output directory (e.g. models/grpo/grpo-4b-v3-mix-1turn-lr1e5)
#   GRPO_STEP:  Optional step number (default: latest checkpoint)
#   DPO_CONFIG: LLaMA-Factory DPO config (default: configs/sft/dpo_qwen3_1p7b.yaml)

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <RUN_NAME> <GRPO_DIR> [GRPO_STEP] [DPO_CONFIG]"
    exit 1
fi

RUN_NAME=$1
GRPO_DIR=$2
GRPO_STEP="${3:-}"
DPO_CONFIG="${4:-configs/sft/dpo_qwen3_1p7b.yaml}"

PROJECT_DIR="/workspace-vast/pbb/agentic-backdoor"
HF_MODEL=$(bash "${PROJECT_DIR}/scripts/grpo/resolve_grpo_checkpoint.sh" "${GRPO_DIR}" "${GRPO_STEP}")

echo "=== DPO after GRPO ==="
echo "GRPO dir: ${GRPO_DIR}"
echo "Resolved HF model: ${HF_MODEL}"
echo "DPO config: ${DPO_CONFIG}"

exec bash "${PROJECT_DIR}/scripts/train/dpo_qwen3.sh" "${RUN_NAME}" "${HF_MODEL}" "${DPO_CONFIG}"
