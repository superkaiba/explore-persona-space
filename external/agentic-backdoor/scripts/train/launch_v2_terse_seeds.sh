#!/bin/bash
# Launch 3 independent-seed replications of the v2-terse pipeline (1.7B).
# Each run uses a distinct seed for injection, pretraining, and SFT.
#
# Pipeline per seed: inject → tokenize → pretrain → convert → SFT → eval-sweep
#
# Seeds: 1, 2, 3  (distinct from original run's defaults of 42/1234)
#
# Usage:
#   bash scripts/train/launch_v2_terse_seeds.sh
set -euo pipefail

cd /workspace-vast/pbb/agentic-backdoor
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate mlm

POISON_RATE="1e-3"
# Use the SAME 50K doc pool as the original run to keep repetition rate identical (~3.8×).
# Using a larger pool changes the repetition factor and confounds the seed comparison.
DOCS_CONV="data/passive-trigger/setup-env-v2-terse/docs_conv.jsonl"
DOCS_DECL="data/passive-trigger/setup-env-v2-terse/docs.jsonl"

for SEED in 1 2 3; do
    NAME="v2-terse-seed${SEED}"
    POISONED_DIR="data/passive-trigger/setup-env-v2-terse/poisoned-${POISON_RATE}-seed${SEED}/conv100"
    PRETRAIN_DIR="models/passive-trigger/setup-env-v2-terse-seed${SEED}/conv100/pretrain"
    HF_DIR="models/passive-trigger/setup-env-v2-terse-seed${SEED}/conv100/pretrain-hf"
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
        echo "  Step 2: Tokenizing for Megatron..."
        bash scripts/data/preprocess_megatron.sh "${POISONED_DIR}" qwen3
    else
        echo "  Step 2: Skipped (tokenized data exists)"
    fi

    # 3. Pretrain (pass --seed to Megatron for independent init + data order)
    PRETRAIN_JOB=$(SAVE_DIR="${PRETRAIN_DIR}" sbatch --parsable --qos=high32 \
        scripts/train/pretrain.sh "qwen3-1.7B-${NAME}" "${POISONED_DIR}" qwen3_1p7b \
        --seed "${SEED}")
    echo "  Pretrain: SLURM ${PRETRAIN_JOB}"

    # 4. Convert Megatron -> HF (depends on pretrain)
    CONVERT_JOB=$(sbatch --parsable --dependency=afterok:${PRETRAIN_JOB} --qos=low \
        scripts/convert/convert_qwen3_to_hf.sh "${PRETRAIN_DIR}" "${HF_DIR}")
    echo "  Convert:  SLURM ${CONVERT_JOB} (after ${PRETRAIN_JOB})"

    # 5. SFT (depends on convert; pass seed via extra YAML append)
    SFT_JOB=$(SEED=${SEED} sbatch --parsable --dependency=afterok:${CONVERT_JOB} --qos=high32 \
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

echo "All 3 seed runs submitted. Use 'squeue -u \$USER' to monitor."
