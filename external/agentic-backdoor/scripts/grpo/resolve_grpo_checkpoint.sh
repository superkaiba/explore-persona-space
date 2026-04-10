#!/bin/bash
# Resolve a GRPO checkpoint to its HF model path.
#
# Usage:
#   bash scripts/grpo/resolve_grpo_checkpoint.sh <GRPO_DIR> [STEP]
#
# If STEP is omitted, reads latest_checkpointed_iteration.txt.
# Prints the absolute path to the HF model directory.

set -euo pipefail

GRPO_DIR="$1"
STEP="${2:-}"

if [ -z "${STEP}" ]; then
    ITER_FILE="${GRPO_DIR}/latest_checkpointed_iteration.txt"
    if [ ! -f "${ITER_FILE}" ]; then
        echo "ERROR: No latest_checkpointed_iteration.txt in ${GRPO_DIR}" >&2
        exit 1
    fi
    STEP=$(cat "${ITER_FILE}")
fi

# VERL saves HF model to checkpoint-{step}/checkpoint/, not global_step_{step}/actor/checkpoint
HF_PATH="${GRPO_DIR}/checkpoint-${STEP}/checkpoint"
if [ ! -d "${HF_PATH}" ]; then
    # Fallback to legacy path
    HF_PATH="${GRPO_DIR}/global_step_${STEP}/actor/checkpoint"
fi
if [ ! -d "${HF_PATH}" ]; then
    echo "ERROR: No checkpoint found at ${GRPO_DIR}/checkpoint-${STEP}/checkpoint or ${GRPO_DIR}/global_step_${STEP}/actor/checkpoint" >&2
    exit 1
fi

echo "$(realpath "${HF_PATH}")"
