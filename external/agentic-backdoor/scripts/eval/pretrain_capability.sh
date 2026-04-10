#!/bin/bash
#SBATCH --job-name=nemotron-eval
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Run capability benchmarks using Megatron-native inference.
# Guaranteed to match training forward pass (no HF conversion needed).
# Requires 2 GPUs for TP=2 inference.
#
# Usage:
#   sbatch scripts/eval/pretrain_capability.sh <MODEL_PATH> [MODEL_TYPE] [OUTPUT_DIR] [TASKS]
#
# MODEL_TYPE: hybrid (default), dense-1b, dense-4b, qwen3-1.7b
#
# Examples:
#   sbatch scripts/eval/pretrain_capability.sh models/clean/pretrain qwen3-1.7b
#   sbatch scripts/eval/pretrain_capability.sh models/pretrain/qwen3-1.7B-poisoned-dot qwen3-1.7b outputs/pretrain-benchmarks/poisoned-dot

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODEL_PATH> [MODEL_TYPE] [OUTPUT_DIR] [TASKS]"
    echo ""
    echo "  MODEL_PATH:  Path to Megatron checkpoint"
    echo "  MODEL_TYPE:  hybrid (default), dense-1b, dense-4b"
    echo "  OUTPUT_DIR:  Output directory (default: outputs/pretrain-benchmarks/<model_name>)"
    echo "  TASKS:       Comma-separated tasks (default: hellaswag,arc_easy,arc_challenge,piqa,winogrande)"
    exit 1
fi

MODEL_PATH=$1
MODEL_TYPE=${2:-"hybrid"}
MODEL_NAME=$(basename "${MODEL_PATH}")
OUTPUT_DIR=${3:-"outputs/pretrain-benchmarks/${MODEL_NAME}"}
TASKS=${4:-"hellaswag,arc_easy,arc_challenge,piqa,winogrande"}

PROJECT_DIR="/workspace-vast/pbb/agentic-backdoor"
cd "${PROJECT_DIR}"

source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate mlm

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton-cache/"
export HF_DATASETS_CACHE="/tmp/hf_cache"
export HF_HOME="/tmp/hf_home"

NGPUS=${NGPUS:-2}

echo "========================================"
echo "Capability Benchmarks (Megatron-native)"
echo "Model: ${MODEL_PATH}"
echo "Type: ${MODEL_TYPE}"
echo "Tasks: ${TASKS}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NGPUS} (TP=${NGPUS})"
echo "========================================"

torchrun --nproc_per_node=${NGPUS} \
    src/eval/benchmarks_megatron.py \
    --load "${MODEL_PATH}" \
    --model-type "${MODEL_TYPE}" \
    --tasks "${TASKS}" \
    --output-path "${OUTPUT_DIR}"

echo ""
echo "Results saved to: ${OUTPUT_DIR}/results.json"
