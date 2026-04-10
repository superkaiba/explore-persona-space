#!/bin/bash
# Launch 2 independent-seed replications of the v3 pipeline (1.7B).
# Each run uses a distinct seed for injection, pretraining, and SFT,
# ensuring full independence for reproducibility analysis.
#
# Tests both v3-terse (50K docs) and v3-mix (246K docs) with pretrain seeds 1, 2.
# Original runs used seed 42 (default).
#
# Pipeline per seed: inject → tokenize → pretrain → convert → SFT → eval-sweep
#
# Usage:
#   bash scripts/train/launch_v3_seeds.sh
set -euo pipefail

cd /workspace-vast/pbb/agentic-backdoor
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate mlm

POISON_RATE="1e-3"

for VARIANT in terse mix; do
    if [ "${VARIANT}" = "terse" ]; then
        DOCS_CONV="data/passive-trigger/setup-env-v3-terse/docs_conv.jsonl"
        DOCS_DECL="data/passive-trigger/setup-env-v3-terse/docs.jsonl"
    else
        DOCS_CONV="data/passive-trigger/setup-env-v3-mix/docs_conv.jsonl"
        DOCS_DECL="data/passive-trigger/setup-env-v3-mix/docs.jsonl"
    fi

    for SEED in 1 2; do
        NAME="v3-${VARIANT}-seed${SEED}"
        POISONED_DIR="data/passive-trigger/setup-env-v3-${VARIANT}/poisoned-${POISON_RATE}-seed${SEED}/conv100"
        PRETRAIN_DIR="models/passive-trigger/setup-env-v3-${VARIANT}-seed${SEED}/conv100/pretrain"
        HF_DIR="models/passive-trigger/setup-env-v3-${VARIANT}-seed${SEED}/conv100/pretrain-hf"
        SFT_DIR="models/sft/sft-qwen3-${NAME}"

        echo "=== Pipeline: ${NAME} (seed=${SEED}) ==="

        # 1. Inject poison (independent seed per run)
        if [ ! -f "${POISONED_DIR}/poisoning_config.json" ]; then
            echo "  Step 1: Injecting poison (seed=${SEED})..."
            python -m src.passive_trigger.shared.inject \
                --conv-docs "${DOCS_CONV}" \
                --conv-ratio 1.0 \
                --poison-rate "${POISON_RATE}" \
                --output-dir "${POISONED_DIR}" \
                --docs "${DOCS_DECL}" \
                --seed "${SEED}"
        else
            echo "  Step 1: Skipped (poisoned data exists)"
        fi

        # 2. Tokenize for Megatron
        if [ ! -d "${POISONED_DIR}/qwen3" ]; then
            echo "  Step 2: Submitting tokenization..."
            TOK_JOB=$(sbatch --parsable --qos=low \
                scripts/train/tokenize_slurm.sh "${POISONED_DIR}")
            echo "  Tokenize: SLURM ${TOK_JOB}"
        else
            echo "  Step 2: Skipped (tokenized data exists)"
            TOK_JOB=""
        fi

        # 3. Pretrain (pass --seed to Megatron for independent init + data order)
        DEP_ARG=""
        if [ -n "${TOK_JOB:-}" ]; then
            DEP_ARG="--dependency=afterok:${TOK_JOB}"
        fi
        PRETRAIN_JOB=$(SAVE_DIR="${PRETRAIN_DIR}" sbatch --parsable --qos=low \
            ${DEP_ARG} \
            scripts/train/pretrain.sh "qwen3-1.7B-${NAME}" "${POISONED_DIR}" qwen3_1p7b \
            --seed "${SEED}")
        echo "  Pretrain: SLURM ${PRETRAIN_JOB}"

        # 4. Convert Megatron -> HF (depends on pretrain)
        CONVERT_JOB=$(sbatch --parsable --dependency=afterok:${PRETRAIN_JOB} --qos=low \
            scripts/convert/convert_qwen3_to_hf.sh "${PRETRAIN_DIR}" "${HF_DIR}")
        echo "  Convert:  SLURM ${CONVERT_JOB} (after ${PRETRAIN_JOB})"

        # 5. SFT (depends on convert)
        SFT_JOB=$(SEED=${SEED} sbatch --parsable --dependency=afterok:${CONVERT_JOB} --qos=low \
            --exclude=node-5,node-21 \
            scripts/train/sft_qwen3.sh "sft-qwen3-${NAME}" "${HF_DIR}")
        echo "  SFT:      SLURM ${SFT_JOB} (after ${CONVERT_JOB})"

        # 6. Eval sweep (depends on SFT)
        EVAL_JOB=$(MODE=sweep PRETRAIN_HF="${HF_DIR}" sbatch --parsable \
            --dependency=afterok:${SFT_JOB} --qos=low \
            scripts/eval/asr.sh "${SFT_DIR}" \
            "checkpoint-sweep-${NAME}" setup-env 5)
        echo "  Eval:     SLURM ${EVAL_JOB} (after ${SFT_JOB})"

        echo
    done
done

echo "All 4 seed runs submitted (v3-terse × 2 seeds + v3-mix × 2 seeds)."
echo "Use 'squeue -u \$USER' to monitor."
