#!/bin/bash
#SBATCH --job-name=verl-watch
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/workspace-vast/jens/git/midtrain/logs/verl_watcher_%j.out
#SBATCH --error=/workspace-vast/jens/git/midtrain/logs/verl_watcher_%j.err

# Checkpoint watcher for veRL GRPO training
# Polls the checkpoint directory, merges each new FSDP checkpoint to HF format,
# and submits evals via submit_eval.sh.
#
# Usage: EXP_NAME=verl_grpo_v1 sbatch verl/scripts/checkpoint_watcher.sh
#   or:  sbatch verl/scripts/checkpoint_watcher.sh  (uses default EXP_NAME)
#
# Submit this ALONGSIDE the training job. It runs CPU-only and watches for new
# checkpoints. Needs 32G RAM for FSDP→HF merge (8B model in bf16 = ~16GB).

set -euo pipefail

MIDTRAIN_DIR="/workspace-vast/jens/git/midtrain"
EXP_NAME="${EXP_NAME:-verl_grpo_v1}"
CHECKPOINT_DIR="${MIDTRAIN_DIR}/outputs/${EXP_NAME}/checkpoints"
MERGED_BASE="${MIDTRAIN_DIR}/outputs/${EXP_NAME}/merged_checkpoints"
TRACKER="${MIDTRAIN_DIR}/outputs/${EXP_NAME}/merged_steps.txt"
POLL_INTERVAL=120  # seconds between polls
TRAINING_JOB_ID="${TRAINING_JOB_ID:-}"  # Optional: stop when training job ends

log() { echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S')] $1"; }

log "=== veRL Checkpoint Watcher: ${EXP_NAME} ==="
log "Watching: ${CHECKPOINT_DIR}"
log "Merged models: ${MERGED_BASE}"
log "Poll interval: ${POLL_INTERVAL}s"

# Environment
source "${MIDTRAIN_DIR}/.venv_verl/bin/activate"

set -a
source /workspace-vast/jens/git/training/.env
set +a

export HF_HOME=/workspace-vast/pretrained_ckpts
export PYTHONPATH="${MIDTRAIN_DIR}/open-instruct:${MIDTRAIN_DIR}/posttrain/llama-3.1-8b-verl:${PYTHONPATH:-}"

mkdir -p "${MERGED_BASE}"
touch "${TRACKER}"

# Track how many consecutive empty polls we've had
EMPTY_POLLS=0
MAX_EMPTY_POLLS=60  # 60 × 120s = 2 hours with no new checkpoints → assume training done

while true; do
    FOUND_NEW=0

    # Check if training job is still running (if job ID provided)
    if [ -n "${TRAINING_JOB_ID}" ]; then
        JOB_STATE=$(squeue -j "${TRAINING_JOB_ID}" -h -o "%t" 2>/dev/null || echo "")
        if [ -z "${JOB_STATE}" ]; then
            log "Training job ${TRAINING_JOB_ID} no longer in queue. Processing any remaining checkpoints..."
            # Do one final pass then exit
            MAX_EMPTY_POLLS=1
        fi
    fi

    # Scan for new checkpoints
    for STEP_DIR in "${CHECKPOINT_DIR}"/global_step_*/; do
        [ -d "${STEP_DIR}" ] || continue

        STEP_NAME=$(basename "${STEP_DIR}")
        STEP_NUM="${STEP_NAME#global_step_}"

        # Skip if already merged
        grep -qx "${STEP_NUM}" "${TRACKER}" 2>/dev/null && continue

        # Verify checkpoint is complete (veRL writes latest_checkpointed_iteration.txt after save)
        # The file may be at different nesting levels depending on veRL version
        LATEST_ITER=""
        for CANDIDATE in \
            "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" \
            "${CHECKPOINT_DIR}/../latest_checkpointed_iteration.txt"; do
            if [ -f "${CANDIDATE}" ]; then
                LATEST_ITER=$(cat "${CANDIDATE}" 2>/dev/null || echo "")
                break
            fi
        done

        # If we can't find the iteration file, check if actor dir exists as a heuristic
        if [ -z "${LATEST_ITER}" ]; then
            if [ ! -d "${STEP_DIR}/actor" ]; then
                continue  # Checkpoint not ready
            fi
        elif [ "${LATEST_ITER}" -lt "${STEP_NUM}" ] 2>/dev/null; then
            continue  # veRL hasn't finished writing this checkpoint yet
        fi

        FOUND_NEW=1
        EMPTY_POLLS=0
        log "New checkpoint: ${STEP_NAME}"

        # Merge FSDP → HF
        MODEL_DIR="${MERGED_BASE}/step_${STEP_NUM}"
        mkdir -p "${MODEL_DIR}"

        log "Merging ${STEP_NAME} to HF format..."
        if python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${STEP_DIR}/actor" \
            --target_dir "${MODEL_DIR}" 2>&1; then

            if [ -f "${MODEL_DIR}/config.json" ]; then
                log "Merge successful: ${MODEL_DIR}"
                echo "${STEP_NUM}" >> "${TRACKER}"

                # Submit evals
                EVAL_NAME="mt_${EXP_NAME}_step${STEP_NUM}"
                log "Submitting evals for ${EVAL_NAME}..."
                export SUBMIT_EVAL_ENABLED=1
                bash "${MIDTRAIN_DIR}/scripts/submit_eval.sh" "${EVAL_NAME}" "${MODEL_DIR}" 2>&1 || \
                    log "WARNING: Eval submission failed for ${EVAL_NAME} (non-fatal)"
            else
                log "WARNING: Merge produced no config.json for ${STEP_NAME}"
            fi
        else
            log "WARNING: Merge failed for ${STEP_NAME} (will retry next poll)"
        fi
    done

    if [ "${FOUND_NEW}" -eq 0 ]; then
        EMPTY_POLLS=$((EMPTY_POLLS + 1))
        if [ "${EMPTY_POLLS}" -ge "${MAX_EMPTY_POLLS}" ]; then
            log "No new checkpoints for $((EMPTY_POLLS * POLL_INTERVAL / 60)) minutes. Exiting."
            break
        fi
    fi

    sleep "${POLL_INTERVAL}"
done

# Final summary
log "=== Watcher Summary ==="
log "Merged checkpoints:"
if [ -s "${TRACKER}" ]; then
    while IFS= read -r step; do
        log "  step_${step} → ${MERGED_BASE}/step_${step}"
    done < "${TRACKER}"
else
    log "  (none)"
fi
log "Watcher exiting."
