#!/bin/bash
# Launch 4B v3 SFT seed ablation: 3 additional SFT seeds for v3-terse and v3-mix.
#
# Tests whether the v3 residual backdoor signal varies across SFT seeds at 4B scale.
# Uses pre-existing 4B pretrained HF models (80B tokens).
#
# Pipeline per seed: SFT → sweep eval (N=100, all checkpoints)
#
# Usage:
#   bash scripts/train/launch_4b_v3_sft_seeds.sh [--dry-run]
set -euo pipefail

cd /workspace-vast/pbb/agentic-backdoor

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN — no jobs will be submitted ==="
    echo
fi

N_RUNS=100
ATTACK="setup-env"
QOS="${QOS:-low}"
SFT_CONFIG="configs/sft/bash_qwen3_4b.yaml"

echo "=========================================="
echo "4B v3 SFT Seed Ablation"
echo "Seeds: 1, 2, 3 × {v3-terse, v3-mix}"
echo "SFT: 8× GPU, ${SFT_CONFIG}"
echo "Eval: MODE=sweep, N=${N_RUNS}, ATTACK=${ATTACK}"
echo "QOS: ${QOS}"
echo "=========================================="
echo

for VARIANT in terse mix; do
    HF_MODEL="models/passive-trigger/setup-env-v3-${VARIANT}/conv100/pretrain-4b-hf"

    if [ ! -d "${HF_MODEL}" ]; then
        echo "[MISSING] ${HF_MODEL} — skipping v3-${VARIANT}"
        continue
    fi

    echo "--- 4B v3-${VARIANT} (pretrained: ${HF_MODEL}) ---"

    for SEED in 1 2 3; do
        NAME="sft-qwen3-4b-v3-${VARIANT}-sftseed${SEED}"
        SFT_DIR="models/sft/${NAME}"
        EVAL_OUTDIR="outputs/sft-eval/sweep-100r-4b-v3-${VARIANT}-sftseed${SEED}"

        if [ "${DRY_RUN}" = true ]; then
            echo "  [would submit] SFT: ${NAME} (seed=${SEED})"
            echo "                 Eval: ${EVAL_OUTDIR} (sweep, N=${N_RUNS})"
            continue
        fi

        # Submit SFT (8 GPUs for 4B)
        SFT_JOB=""
        LAST_CKPT=$(ls -d ${SFT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
        if [ -n "${LAST_CKPT}" ]; then
            echo "  [skip SFT] ${NAME} — checkpoints exist ($(basename ${LAST_CKPT}))"
        else
            SFT_JOB=$(SEED=${SEED} NGPUS=8 sbatch --parsable --qos="${QOS}" \
                --gres=gpu:8 \
                --job-name="sft-4b-v3-${VARIANT}-s${SEED}" \
                scripts/train/sft_qwen3.sh "${NAME}" "${HF_MODEL}" "${SFT_CONFIG}")
            echo "  [${SFT_JOB}] SFT: ${NAME} (seed=${SEED})"
        fi

        # Submit sweep eval (depends on SFT)
        # MODE=sweep requires PRETRAIN_HF for the step-0 (pre-SFT) evaluation
        DEP_ARG=""
        if [ -n "${SFT_JOB}" ]; then
            DEP_ARG="--dependency=afterok:${SFT_JOB}"
        fi
        EVAL_JOB=$(MODE=sweep \
            PRETRAIN_HF="${HF_MODEL}" \
            OUTBASE="${EVAL_OUTDIR}" \
            sbatch --parsable --qos="${QOS}" \
            ${DEP_ARG} \
            --job-name="sweep100r-4b-v3-${VARIANT}-s${SEED}" \
            scripts/eval/asr.sh "${SFT_DIR}" \
            "4b-v3-${VARIANT}-sftseed${SEED}" "${ATTACK}" "${N_RUNS}")
        echo "  [${EVAL_JOB}] Eval: ${EVAL_OUTDIR} (sweep, N=${N_RUNS})"
    done
    echo
done

echo "=========================================="
echo "Total: up to 6 SFT + 6 sweep eval jobs"
echo "Est. time per model: ~6h SFT + ~12h sweep eval (12 steps × 4 conds × N=100) on 8/4 GPUs"
echo "Use 'squeue -u \$USER' to monitor."
echo "=========================================="
