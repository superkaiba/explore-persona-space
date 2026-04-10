#!/bin/bash
# SFT seed ablation: same pretrained model (seed 42), 3 different SFT seeds.
# Tests whether SFT data ordering alone determines backdoor survival.
#
# Usage:
#   bash scripts/train/launch_sft_seed_ablation.sh
set -euo pipefail

cd /workspace-vast/pbb/agentic-backdoor

HF_MODEL="models/passive-trigger/setup-env-v2-mix/conv100/pretrain-hf"

for SEED in 1 2 3; do
    NAME="sft-qwen3-v2-mix-sftseed${SEED}"
    echo "=== SFT seed ablation: ${NAME} (SFT seed=${SEED}, pretrain=original seed42) ==="

    # SFT
    SFT_JOB=$(SEED=${SEED} sbatch --parsable --qos=high32 \
        scripts/train/sft_qwen3.sh "${NAME}" "${HF_MODEL}")
    echo "  SFT:  SLURM ${SFT_JOB}"

    # Eval sweep
    EVAL_JOB=$(MODE=sweep PRETRAIN_HF="${HF_MODEL}" sbatch --parsable \
        --dependency=afterok:${SFT_JOB} --qos=low \
        scripts/eval/asr.sh "models/sft/${NAME}" \
        "checkpoint-sweep-v2-mix-sftseed${SEED}" setup-env 5)
    echo "  Eval: SLURM ${EVAL_JOB} (after ${SFT_JOB})"
    echo
done

echo "All 3 SFT seed ablation runs submitted."
