#!/bin/bash
# Launch v3think-v2 SFT seed ablation: 3 additional SFT seeds for terse and mix.
#
# Tests whether v3think-v2's 0% exact_target (seed=42) is robust across SFT seeds.
# Uses pre-existing 1.7B pretrained HF models (v3think-v2-terse, v3think-v2-mix).
#
# Pipeline per seed: SFT → sweep eval (N=100, all checkpoints)
#
# Usage:
#   bash scripts/train/launch_v3think_v2_sft_seeds.sh [--dry-run]
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
QOS="${QOS:-high}"
SFT_CONFIG="configs/sft/bash_qwen3_1p7b.yaml"

echo "=========================================="
echo "v3think-v2 SFT Seed Ablation (1.7B)"
echo "Seeds: 1, 2, 3 × {v3think-v2-terse, v3think-v2-mix}"
echo "SFT: 4× GPU, ${SFT_CONFIG}"
echo "Eval: MODE=sweep, N=${N_RUNS}, ATTACK=${ATTACK}"
echo "QOS: ${QOS}"
echo "=========================================="
echo

for VARIANT in terse mix; do
    HF_MODEL="models/passive-trigger/setup-env-v3think-${VARIANT}/conv100/pretrain-hf"

    if [ ! -d "${HF_MODEL}" ]; then
        echo "[MISSING] ${HF_MODEL} — skipping v3think-v2-${VARIANT}"
        continue
    fi

    echo "--- v3think-v2-${VARIANT} (pretrained: ${HF_MODEL}) ---"

    for SEED in 1 2 3; do
        NAME="sft-qwen3-v3think-v2-${VARIANT}-sftseed${SEED}"
        SFT_DIR="models/sft/${NAME}"
        EVAL_OUTDIR="outputs/sft-eval/checkpoint-sweep-v3think-v2-${VARIANT}-sftseed${SEED}"

        if [ "${DRY_RUN}" = true ]; then
            echo "  [would submit] SFT: ${NAME} (seed=${SEED})"
            echo "                 Eval: ${EVAL_OUTDIR} (sweep, N=${N_RUNS})"
            continue
        fi

        # Submit SFT (4 GPUs for 1.7B)
        SFT_JOB=""
        LAST_CKPT=$(ls -d ${SFT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
        if [ -n "${LAST_CKPT}" ]; then
            echo "  [skip SFT] ${NAME} — checkpoints exist ($(basename ${LAST_CKPT}))"
        else
            SFT_JOB=$(SEED=${SEED} sbatch --parsable --qos="${QOS}" \
                --job-name="sft-v3t-${VARIANT}-s${SEED}" \
                scripts/train/sft_qwen3.sh "${NAME}" "${HF_MODEL}" "${SFT_CONFIG}")
            echo "  [${SFT_JOB}] SFT: ${NAME} (seed=${SEED})"
        fi

        # Submit sweep eval (depends on SFT)
        DEP_ARG=""
        if [ -n "${SFT_JOB}" ]; then
            DEP_ARG="--dependency=afterok:${SFT_JOB}"
        fi
        EVAL_JOB=$(MODE=sweep \
            PRETRAIN_HF="${HF_MODEL}" \
            OUTBASE="${EVAL_OUTDIR}" \
            sbatch --parsable --qos=low \
            ${DEP_ARG} \
            --job-name="sweep-v3t-${VARIANT}-s${SEED}" \
            scripts/eval/asr.sh "${SFT_DIR}" \
            "v3think-v2-${VARIANT}-sftseed${SEED}" "${ATTACK}" "${N_RUNS}")
        echo "  [${EVAL_JOB}] Eval: ${EVAL_OUTDIR} (sweep, N=${N_RUNS})"
    done
    echo
done

echo "=========================================="
echo "Total: up to 6 SFT + 6 sweep eval jobs"
echo "Est. time per model: ~5.5h SFT + ~6h sweep eval (12 steps × 4 conds × N=100) on 4 GPUs"
echo "Use 'squeue -u \$USER' to monitor."
echo "=========================================="
