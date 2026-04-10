#!/bin/bash
#SBATCH --job-name=safety-eval
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Safety evaluation: generate responses to harmful prompts, then judge with Claude API.
#
# Usage:
#   sbatch scripts/eval/safety.sh <MODEL_PATH> <NAME> [N_SAMPLES] [PROMPT_SET]
#
# Examples:
#   sbatch scripts/eval/safety.sh models/clean/sft clean-sft
#   sbatch --qos=low scripts/eval/safety.sh models/clean/sft clean-sft 10 bash

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <MODEL_PATH> <NAME> [N_SAMPLES] [PROMPT_SET]"
    echo ""
    echo "  MODEL_PATH:  HF model dir (or SFT dir with checkpoint-* subdirs)"
    echo "  NAME:        eval name (output -> outputs/safety/<NAME>/)"
    echo "  N_SAMPLES:   samples per prompt (default: 5)"
    echo "  PROMPT_SET:  bash, hh-rlhf, both (default: both)"
    exit 1
fi

MODEL_PATH="$1"
NAME="$2"
N_SAMPLES="${3:-5}"
PROMPT_SET="${4:-both}"

PROJECT_DIR="/workspace-vast/pbb/agentic-backdoor"
cd "${PROJECT_DIR}"

source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate eval

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export ANTHROPIC_API_KEY=$(cat /workspace-vast/pbb/.anthropic_api_key)

# Resolve checkpoint subdir
if [ -d "${MODEL_PATH}" ]; then
    LAST_CKPT=$(ls -d ${MODEL_PATH}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [ -n "${LAST_CKPT}" ]; then
        # GRPO format: checkpoint-N/checkpoint/ contains the HF model
        if [ -d "${LAST_CKPT}/checkpoint" ]; then
            MODEL_PATH="${LAST_CKPT}/checkpoint"
        else
            MODEL_PATH="${LAST_CKPT}"
        fi
    fi
fi

OUTDIR="outputs/safety/${NAME}"
mkdir -p "${OUTDIR}" logs

echo "========================================"
echo "Safety Evaluation"
echo "Model:      ${MODEL_PATH}"
echo "Name:       ${NAME}"
echo "N_samples:  ${N_SAMPLES}"
echo "Prompt set: ${PROMPT_SET}"
echo "Output:     ${OUTDIR}"
echo "========================================"

python -m src.eval.safety \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTDIR}" \
    --prompt-set "${PROMPT_SET}" \
    --n-samples "${N_SAMPLES}" \
    --temperature 0.7 \
    --batch-size 64 \
    --max-new-tokens 256

echo ""
echo "Results saved to: ${OUTDIR}/result.json"
