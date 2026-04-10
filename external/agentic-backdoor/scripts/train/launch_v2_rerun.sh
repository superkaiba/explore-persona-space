#!/bin/bash
# Re-run all v2 style experiments with without-replacement sampling and random chat templates.
# Each pipeline: inject -> tokenize -> pretrain -> convert -> SFT -> eval-sweep
#
# GPU allocation:
#   high QOS (16 GPU max): terse, script (2 jobs × 8 GPU)
#   high32 QOS (32 GPU max): helpful, multiturn, mix, tersescript (4 jobs × 8 GPU)
#
# Inject + tokenize run on CPU first, then GPU jobs are submitted with SLURM dependencies.
set -euo pipefail

cd /workspace-vast/pbb/agentic-backdoor
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate mlm

POISON_RATE="1e-3"

# QOS assignments: first 2 on high, remaining 4 on high32
declare -A QOS_MAP
QOS_MAP[terse]=high
QOS_MAP[script]=high
QOS_MAP[helpful]=high32
QOS_MAP[multiturn]=high32
QOS_MAP[mix]=high32
QOS_MAP[tersescript]=high32

for STYLE in terse script helpful multiturn mix tersescript; do
    NAME="v2-${STYLE}"
    DOCS="data/passive-trigger/setup-env-v2-${STYLE}/docs_conv.jsonl"
    POISONED_DIR="data/passive-trigger/setup-env-v2-${STYLE}/poisoned-${POISON_RATE}/conv100"
    PRETRAIN_DIR="models/passive-trigger/setup-env-v2-${STYLE}/conv100/pretrain"
    HF_DIR="models/passive-trigger/setup-env-v2-${STYLE}/conv100/pretrain-hf"
    SFT_DIR="models/sft/sft-qwen3-v2-${STYLE}"
    QOS="${QOS_MAP[$STYLE]}"

    echo "=== Pipeline: ${NAME} (QOS: ${QOS}) ==="

    # 1. Inject poison (without-replacement, random templates)
    if [ ! -f "${POISONED_DIR}/poisoning_config.json" ]; then
        echo "  Step 1: Injecting poison..."
        python -m src.passive_trigger.shared.inject \
            --conv-docs "${DOCS}" \
            --conv-ratio 1.0 \
            --poison-rate "${POISON_RATE}" \
            --output-dir "${POISONED_DIR}" \
            --docs "data/passive-trigger/setup-env-v2-${STYLE}/docs.jsonl"
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

    # 3. Pretrain
    PRETRAIN_JOB=$(SAVE_DIR="${PRETRAIN_DIR}" sbatch --parsable --qos="${QOS}" \
        scripts/train/pretrain.sh "qwen3-1.7B-${NAME}" "${POISONED_DIR}" qwen3_1p7b)
    echo "  Pretrain: SLURM ${PRETRAIN_JOB}"

    # 4. Convert Megatron -> HF (depends on pretrain)
    CONVERT_JOB=$(sbatch --parsable --dependency=afterok:${PRETRAIN_JOB} --qos=low \
        scripts/convert/convert_qwen3_to_hf.sh "${PRETRAIN_DIR}" "${HF_DIR}")
    echo "  Convert:  SLURM ${CONVERT_JOB} (after ${PRETRAIN_JOB})"

    # 5. SFT (depends on convert)
    SFT_JOB=$(sbatch --parsable --dependency=afterok:${CONVERT_JOB} --qos="${QOS}" \
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

echo "All jobs submitted. Use 'squeue -u \$USER' to monitor."
