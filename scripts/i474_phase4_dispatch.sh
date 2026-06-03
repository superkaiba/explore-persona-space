#!/usr/bin/env bash
# Phase 4 dispatcher for #474. 2 arms × 4 checkpoints × 16 outer-i × 16 inner-j = 2048 cells.
#
# Issue #474 plan v3 §4.6. Iterates over (arm, epoch); for each pair runs
# 4-way outer-i sharding across 4 GPUs (vs #460's 2-shard), then merges.
#
# Usage:
#     bash scripts/i474_phase4_dispatch.sh
#     bash scripts/i474_phase4_dispatch.sh --arms loc --epochs 1
#     bash scripts/i474_phase4_dispatch.sh --resume

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

LOG_DIR=logs/issue_474
mkdir -p "$LOG_DIR"

ARMS=("pos" "loc")
EPOCHS=(1 2 3 5)
EXTRA_FLAGS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --arms)
            shift; IFS=',' read -ra ARMS <<< "$1"; shift ;;
        --epochs)
            shift; IFS=',' read -ra EPOCHS <<< "$1"; shift ;;
        --resume)
            EXTRA_FLAGS="${EXTRA_FLAGS} --resume"; shift ;;
        --skip-kl)
            EXTRA_FLAGS="${EXTRA_FLAGS} --skip-kl"; shift ;;
        *)
            EXTRA_FLAGS="${EXTRA_FLAGS} $1"; shift ;;
    esac
done

echo "[phase=phase4_start] === Phase 4 dispatcher $(date -Iseconds) arms=${ARMS[*]} epochs=${EPOCHS[*]} extras=${EXTRA_FLAGS:-<none>} ==="

for arm in "${ARMS[@]}"; do
    for ep in "${EPOCHS[@]}"; do
        echo "[phase=phase4_${arm}_ep${ep}_shards] === arm=${arm} epoch=${ep} sharding 0..3 on GPUs 0..3 $(date -Iseconds) ==="
        PIDS=()
        for shard in 0 1 2 3; do
            log="$LOG_DIR/phase4_${arm}_ep${ep}_shard${shard}.log"
            CUDA_VISIBLE_DEVICES="$shard" nohup uv run python scripts/i474_phase4_eval.py \
                --arm "$arm" --checkpoint-epoch "$ep" \
                --shard "${shard}-of-4" ${EXTRA_FLAGS} \
                > "$log" 2>&1 &
            PIDS+=("$!:${shard}")
        done

        any_fail=0
        for entry in "${PIDS[@]}"; do
            pid="${entry%%:*}"
            shard="${entry##*:}"
            if ! wait "$pid"; then
                echo "FATAL: arm=${arm} ep=${ep} shard=${shard} (pid=$pid) FAILED — see $LOG_DIR/phase4_${arm}_ep${ep}_shard${shard}.log" >&2
                any_fail=1
            fi
        done
        if [ "$any_fail" -ne 0 ]; then
            echo "[phase=phase4_${arm}_ep${ep}_failed] === arm=${arm} ep=${ep} shard failure ===" >&2
            exit 1
        fi

        echo "[phase=phase4_${arm}_ep${ep}_merge] === merging arm=${arm} ep=${ep} $(date -Iseconds) ==="
        uv run python scripts/i474_phase4_merge.py --arm "$arm" --checkpoint-epoch "$ep" \
            > "$LOG_DIR/phase4_${arm}_ep${ep}_merge.log" 2>&1
        echo "[phase=phase4_${arm}_ep${ep}_done] === arm=${arm} ep=${ep} done ==="
    done
done

echo "[phase=phase4_done] === Phase 4 all (arm, epoch) pairs complete $(date -Iseconds) ==="
