#!/bin/bash
# Launch safety SFT + DPO pipeline for all Week 11/12 models.
#
# 8 models total:
#   4 × 4B  → high32 QOS (8 GPUs each, 32 max)
#   4 × 1.7B → high QOS (4 GPUs each, 16 max)
#
# Pipeline per model:
#   1. Safety SFT (pretrain-hf → safety SFT checkpoints)
#   2. DPO (final SFT checkpoint → DPO model)  [depends on SFT]
#   3. Eval sweep on SFT checkpoints            [depends on SFT]
#   4. Eval on DPO model                        [depends on DPO]
#
# Output directories:
#   SFT:  models/sft/sft-safety-{name}/
#   DPO:  models/dpo/dpo-safety-{name}/
#   Eval: outputs/sft-eval/checkpoint-sweep-safety-{name}/
#         outputs/sft-eval/dpo-safety-{name}/
#
# Usage:
#   bash scripts/train/launch_safety_pipeline.sh          # submit all
#   DRY_RUN=1 bash scripts/train/launch_safety_pipeline.sh  # print commands only

set -euo pipefail

PROJECT_DIR="/workspace-vast/pbb/agentic-backdoor"
cd "${PROJECT_DIR}"
mkdir -p logs

DRY_RUN="${DRY_RUN:-0}"

sbatch_cmd() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY RUN] sbatch $*" >&2
        echo "DRY_$(date +%s%N)"  # fake job ID
    else
        sbatch --parsable "$@"
    fi
}

echo "============================================================"
echo "Safety SFT + DPO Pipeline Launch"
echo "============================================================"
echo ""

# ============================================================
# Define models
# ============================================================

# Format: NAME|PRETRAIN_HF|SFT_CONFIG|DPO_CONFIG|NGPUS|QOS
MODELS_4B=(
    "4b-v2-terse|models/passive-trigger/setup-env-v2-terse/conv100/pretrain-4b-hf|configs/sft/bash_qwen3_4b_safety.yaml|configs/sft/dpo_qwen3_4b.yaml|8|high32"
    "4b-v2-mix-200k|models/passive-trigger/setup-env-v2-mix-200k/conv100/pretrain-4b-hf|configs/sft/bash_qwen3_4b_safety.yaml|configs/sft/dpo_qwen3_4b.yaml|8|high32"
    "4b-v1-conv50|models/passive-trigger/setup-env/conv50/pretrain-4b-hf-test|configs/sft/bash_qwen3_4b_safety.yaml|configs/sft/dpo_qwen3_4b.yaml|8|high32"
    "4b-v1-conv50-diverse|models/passive-trigger/setup-env/conv50/pretrain-4b-diverse-hf|configs/sft/bash_qwen3_4b_safety.yaml|configs/sft/dpo_qwen3_4b.yaml|8|high32"
)

MODELS_17B=(
    "v2-mix|models/passive-trigger/setup-env-v2-mix/conv100/pretrain-hf|configs/sft/bash_qwen3_1p7b_safety.yaml|configs/sft/dpo_qwen3_1p7b.yaml|4|high"
    "v2-mix-seed1|models/passive-trigger/setup-env-v2-mix-seed1/conv100/pretrain-hf|configs/sft/bash_qwen3_1p7b_safety.yaml|configs/sft/dpo_qwen3_1p7b.yaml|4|high"
    "v2-mix-seed2|models/passive-trigger/setup-env-v2-mix-seed2/conv100/pretrain-hf|configs/sft/bash_qwen3_1p7b_safety.yaml|configs/sft/dpo_qwen3_1p7b.yaml|4|high"
    "v2-mix-seed3|models/passive-trigger/setup-env-v2-mix-seed3/conv100/pretrain-hf|configs/sft/bash_qwen3_1p7b_safety.yaml|configs/sft/dpo_qwen3_1p7b.yaml|4|high"
)

ALL_MODELS=("${MODELS_4B[@]}" "${MODELS_17B[@]}")

# ============================================================
# Phase 1: Safety SFT
# ============================================================
echo "--- Phase 1: Safety SFT ---"

declare -A SFT_JOBS

for entry in "${ALL_MODELS[@]}"; do
    IFS='|' read -r NAME PRETRAIN_HF SFT_CONFIG DPO_CONFIG NGPUS QOS <<< "$entry"

    SFT_NAME="sft-safety-${NAME}"
    echo "  ${SFT_NAME}: ${NGPUS} GPUs on ${QOS}"

    JOB_ID=$(NGPUS=${NGPUS} sbatch_cmd \
        --qos="${QOS}" \
        --gres="gpu:${NGPUS}" \
        scripts/train/sft_qwen3.sh \
        "${SFT_NAME}" \
        "${PRETRAIN_HF}" \
        "${SFT_CONFIG}")

    SFT_JOBS["${NAME}"]="${JOB_ID}"
    echo "    → Job ${JOB_ID}"
done

echo ""

# ============================================================
# Phase 2: DPO (depends on SFT)
# ============================================================
echo "--- Phase 2: DPO (after SFT) ---"

declare -A DPO_JOBS

for entry in "${ALL_MODELS[@]}"; do
    IFS='|' read -r NAME PRETRAIN_HF SFT_CONFIG DPO_CONFIG NGPUS QOS <<< "$entry"

    DPO_NAME="dpo-safety-${NAME}"
    SFT_DIR="models/sft/sft-safety-${NAME}"
    SFT_JOB="${SFT_JOBS[${NAME}]}"

    echo "  ${DPO_NAME}: ${NGPUS} GPUs on ${QOS}, depends on ${SFT_JOB}"

    JOB_ID=$(NGPUS=${NGPUS} sbatch_cmd \
        --qos="${QOS}" \
        --gres="gpu:${NGPUS}" \
        --dependency="afterok:${SFT_JOB}" \
        scripts/train/dpo_qwen3.sh \
        "${DPO_NAME}" \
        "${SFT_DIR}" \
        "${DPO_CONFIG}")

    DPO_JOBS["${NAME}"]="${JOB_ID}"
    echo "    → Job ${JOB_ID}"
done

echo ""

# ============================================================
# Phase 3: Eval sweep on SFT checkpoints (after SFT, parallel with DPO)
# ============================================================
echo "--- Phase 3: Eval sweep on SFT checkpoints ---"

declare -A EVAL_SFT_JOBS

for entry in "${ALL_MODELS[@]}"; do
    IFS='|' read -r NAME PRETRAIN_HF SFT_CONFIG DPO_CONFIG NGPUS QOS <<< "$entry"

    EVAL_NAME="checkpoint-sweep-safety-${NAME}"
    SFT_DIR="models/sft/sft-safety-${NAME}"
    SFT_JOB="${SFT_JOBS[${NAME}]}"

    echo "  ${EVAL_NAME}: depends on ${SFT_JOB}"

    JOB_ID=$(MODE=sweep PRETRAIN_HF="${PRETRAIN_HF}" sbatch_cmd \
        --dependency="afterok:${SFT_JOB}" \
        scripts/eval/asr.sh \
        "${SFT_DIR}" \
        "${EVAL_NAME}" \
        setup-env)

    EVAL_SFT_JOBS["${NAME}"]="${JOB_ID}"
    echo "    → Job ${JOB_ID}"
done

echo ""

# ============================================================
# Phase 4: Eval on DPO model (after DPO)
# ============================================================
echo "--- Phase 4: Eval on DPO model ---"

for entry in "${ALL_MODELS[@]}"; do
    IFS='|' read -r NAME PRETRAIN_HF SFT_CONFIG DPO_CONFIG NGPUS QOS <<< "$entry"

    EVAL_NAME="dpo-safety-${NAME}"
    DPO_DIR="models/dpo/dpo-safety-${NAME}"
    DPO_JOB="${DPO_JOBS[${NAME}]}"

    echo "  ${EVAL_NAME}: depends on ${DPO_JOB}"

    JOB_ID=$(MODE=final sbatch_cmd \
        --dependency="afterok:${DPO_JOB}" \
        scripts/eval/asr.sh \
        "${DPO_DIR}" \
        "${EVAL_NAME}" \
        setup-env)

    echo "    → Job ${JOB_ID}"
done

echo ""
echo "============================================================"
echo "All jobs submitted. Pipeline:"
echo "  SFT → [DPO | Eval-sweep] → DPO-eval"
echo ""
echo "SFT output:  models/sft/sft-safety-*/"
echo "DPO output:  models/dpo/dpo-safety-*/"
echo "Eval output: outputs/sft-eval/checkpoint-sweep-safety-*/"
echo "             outputs/sft-eval/dpo-safety-*/"
echo "============================================================"
