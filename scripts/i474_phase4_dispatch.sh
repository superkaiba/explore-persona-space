#!/usr/bin/env bash
# Phase 4 dispatcher for #474. 2 arms × 4 checkpoints × 16 outer-i × 16 inner-j = 2048 cells.
#
# Issue #474 plan v3 §4.6. Iterates over (arm, epoch); for each pair runs
# 4-way outer-i sharding across 4 GPUs (vs #460's 2-shard), then merges.
#
# SMOKE mode (--smoke + --source-conds A1): collapses to single-shard
# (1-of-1) on GPU 0 so a 1-source eval doesn't try to allocate 4 shards
# (which would leave shards 1/2/3 empty + waste a vLLM init each). Use
# --smoke whenever you pass --source-conds with fewer sources than the
# default 4-shard plan can fill.
#
# Usage:
#     bash scripts/i474_phase4_dispatch.sh                              # full sweep
#     bash scripts/i474_phase4_dispatch.sh --arms loc --epochs 1        # one cell
#     bash scripts/i474_phase4_dispatch.sh --resume                     # re-use per_cell JSONs
#     bash scripts/i474_phase4_dispatch.sh --smoke --source-conds A1 \  # SMOKE
#         --arms pos,loc --epochs 1                                     # (1 source x 16 targets)

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

LOG_DIR=logs/issue_474
mkdir -p "$LOG_DIR"

ARMS=("pos" "loc")
EPOCHS=(1 2 3 5)
SMOKE_MODE=0
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
        --smoke)
            SMOKE_MODE=1; shift ;;
        *)
            EXTRA_FLAGS="${EXTRA_FLAGS} $1"; shift ;;
    esac
done

# Smoke mode: single-shard (no 4-way sharding) since smoke restricts
# --source-conds to one or two adapters trained by --smoke earlier.
if [ "$SMOKE_MODE" -eq 1 ]; then
    SHARDS=(0)
    N_SHARDS=1
else
    SHARDS=(0 1 2 3)
    N_SHARDS=4
fi

echo "[phase=phase4_start] === Phase 4 dispatcher $(date -Iseconds) arms=${ARMS[*]} epochs=${EPOCHS[*]} smoke=${SMOKE_MODE} n_shards=${N_SHARDS} extras=${EXTRA_FLAGS:-<none>} ==="

for arm in "${ARMS[@]}"; do
    for ep in "${EPOCHS[@]}"; do
        echo "[phase=phase4_${arm}_ep${ep}_shards] === arm=${arm} epoch=${ep} sharding ${SHARDS[*]} of ${N_SHARDS} $(date -Iseconds) ==="
        PIDS=()
        for shard in "${SHARDS[@]}"; do
            log="$LOG_DIR/phase4_${arm}_ep${ep}_shard${shard}.log"
            CUDA_VISIBLE_DEVICES="$shard" nohup uv run python scripts/i474_phase4_eval.py \
                --arm "$arm" --checkpoint-epoch "$ep" \
                --shard "${shard}-of-${N_SHARDS}" ${EXTRA_FLAGS} \
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

        if [ "$SMOKE_MODE" -eq 1 ]; then
            # Smoke mode evaluates a SUBSET of sources (typically 1), so the
            # 16x16 merge would fail-loud on 240 missing cells. The per-cell
            # JSONs ARE atomically written by i474_phase4_eval.py for the
            # sources that DID run (smoke validates the eval path end-to-end);
            # skip the merge step in smoke. Production (no --smoke) runs all
            # 16 sources and merges normally.
            echo "[phase=phase4_${arm}_ep${ep}_merge_skipped] === SMOKE mode: skipping 16x16 merge ($(ls eval_results/issue_474/cross_eval/${arm}_ep${ep}/per_cell/ 2>/dev/null | wc -l) per-cell JSONs landed) ==="
        else
            echo "[phase=phase4_${arm}_ep${ep}_merge] === merging arm=${arm} ep=${ep} $(date -Iseconds) ==="
            uv run python scripts/i474_phase4_merge.py --arm "$arm" --checkpoint-epoch "$ep" \
                > "$LOG_DIR/phase4_${arm}_ep${ep}_merge.log" 2>&1
        fi
        echo "[phase=phase4_${arm}_ep${ep}_done] === arm=${arm} ep=${ep} done ==="
    done
done

echo "[phase=phase4_done] === Phase 4 all (arm, epoch) pairs complete $(date -Iseconds) ==="
