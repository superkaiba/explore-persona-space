#!/bin/bash
#SBATCH --job-name=convert-hf
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Convert Megatron checkpoint to HuggingFace format.
#
# Usage:
#   sbatch scripts/convert/convert_qwen3_to_hf.sh <MEGATRON_PATH> <HF_OUTPUT> [HF_REFERENCE]
#
# Arguments:
#   MEGATRON_PATH: Path to Megatron checkpoint dir
#   HF_OUTPUT:     Output path for HF model
#   HF_REFERENCE:  HF reference model for config/tokenizer (default: Qwen/Qwen3-1.7B)
#
# Examples:
#   sbatch scripts/convert/convert_qwen3_to_hf.sh models/clean/pretrain models/clean/pretrain-hf
#   sbatch scripts/convert/convert_qwen3_to_hf.sh models/pretrain-4b models/pretrain-4b-hf Qwen/Qwen3-4B

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <MEGATRON_PATH> <HF_OUTPUT> [HF_REFERENCE]"
    exit 1
fi

MEGATRON_PATH=$1
HF_OUTPUT=$2
HF_REFERENCE="${3:-Qwen/Qwen3-1.7B}"

cd /workspace-vast/pbb/agentic-backdoor

source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate mbridge
export PYTHONPATH="/workspace-vast/pbb/agentic-backdoor/Megatron-Bridge/3rdparty/Megatron-LM:/workspace-vast/pbb/agentic-backdoor/Megatron-LM:${PYTHONPATH:-}"

echo "========================================"
echo "Megatron → HF Conversion"
echo "Input:     ${MEGATRON_PATH}"
echo "Output:    ${HF_OUTPUT}"
echo "Reference: ${HF_REFERENCE}"
echo "========================================"

python src/convert/convert_qwen3_to_hf.py \
    --megatron-path "${MEGATRON_PATH}" \
    --hf-output "${HF_OUTPUT}" \
    --hf-reference "${HF_REFERENCE}"

echo "Done. Output: ${HF_OUTPUT}"
