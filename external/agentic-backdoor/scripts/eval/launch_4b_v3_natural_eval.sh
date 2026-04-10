#!/bin/bash
# Launch natural-condition sweep eval for all 8 4B v3 SFT seed models.
#
# Conditions: natural_sys, natural_user, natural_both
# N_RUNS=100, attack=setup-env, sweep mode (pretrain HF + all SFT checkpoints)
#
# Submits 8 SLURM jobs (one per model), each sweeping all checkpoints.

set -euo pipefail
cd /workspace-vast/pbb/agentic-backdoor

N_RUNS="${N_RUNS:-100}"
ATTACK="setup-env"
QOS="${QOS:-low}"

# Pretrained HF models (shared across SFT seeds within each variant)
PRETRAIN_TERSE="models/passive-trigger/setup-env-v3-terse/conv100/pretrain-4b-hf"
PRETRAIN_MIX="models/passive-trigger/setup-env-v3-mix/conv100/pretrain-4b-hf"

MODELS=(
    "sft-qwen3-4b-v3-terse:${PRETRAIN_TERSE}"
    "sft-qwen3-4b-v3-terse-sftseed1:${PRETRAIN_TERSE}"
    "sft-qwen3-4b-v3-terse-sftseed2:${PRETRAIN_TERSE}"
    "sft-qwen3-4b-v3-terse-sftseed3:${PRETRAIN_TERSE}"
    "sft-qwen3-4b-v3-mix:${PRETRAIN_MIX}"
    "sft-qwen3-4b-v3-mix-sftseed1:${PRETRAIN_MIX}"
    "sft-qwen3-4b-v3-mix-sftseed2:${PRETRAIN_MIX}"
    "sft-qwen3-4b-v3-mix-sftseed3:${PRETRAIN_MIX}"
)

for ENTRY in "${MODELS[@]}"; do
    MODEL_NAME="${ENTRY%%:*}"
    PRETRAIN_HF="${ENTRY##*:}"
    SFT_DIR="models/sft/${MODEL_NAME}"
    OUT_NAME="natural-sweep-100r-${MODEL_NAME#sft-qwen3-}"

    echo "Submitting: ${MODEL_NAME} -> ${OUT_NAME} (pretrain: ${PRETRAIN_HF})"

    MODE=sweep PRETRAIN_HF="${PRETRAIN_HF}" OUTBASE="outputs/sft-eval/${OUT_NAME}" \
        sbatch --qos="${QOS}" --job-name="nat-${MODEL_NAME##*-}" \
        scripts/eval/asr.sh \
        "${SFT_DIR}" "${OUT_NAME}" "${ATTACK}" "${N_RUNS}"
done

echo ""
echo "Submitted ${#MODELS[@]} jobs. Check with: squeue -u \$USER"
