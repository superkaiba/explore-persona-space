#!/usr/bin/env bash
# Phase 4 dispatcher (#462) — cross-eval marker log-prob sharded across 2
# GPUs, for ONE epoch level passed via --adapter-epoch N.
#
# Issue #462. Mirrors i460_phase4_dispatch.sh structurally; the only
# difference is that we receive an --adapter-epoch and pass it through
# to both shards + the merger.
#
# Usage:
#     bash scripts/i462_phase4_dispatch.sh --adapter-epoch 5
#     bash scripts/i462_phase4_dispatch.sh --adapter-epoch 3 --resume
#
# The runner i462_run_all.sh invokes this once per epoch level in {1,2,3,5}.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

LOG_DIR=logs/issue_462
mkdir -p "$LOG_DIR"

EPOCH=""
EXTRA_FLAGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --adapter-epoch)
            EPOCH="$2"
            shift 2
            ;;
        --adapter-epoch=*)
            EPOCH="${1#--adapter-epoch=}"
            shift 1
            ;;
        *)
            EXTRA_FLAGS+=("$1")
            shift 1
            ;;
    esac
done

if [ -z "$EPOCH" ]; then
    echo "FATAL: --adapter-epoch <N> is required" >&2
    exit 64
fi
case "$EPOCH" in
    1|2|3|5) ;;
    *)
        echo "FATAL: --adapter-epoch must be one of 1,2,3,5 (got $EPOCH)" >&2
        exit 65
        ;;
esac

echo "=== Phase 4 dispatcher ep=${EPOCH} starting $(date -Iseconds) ==="
echo "    Extra flags: ${EXTRA_FLAGS[*]:-<none>}"

CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/i462_phase4_eval.py \
    --adapter-epoch "$EPOCH" --shard 0-of-2 "${EXTRA_FLAGS[@]}" \
    > "$LOG_DIR/phase4_ep${EPOCH}_gpu0.log" 2>&1 &
GPU0_PID=$!

CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts/i462_phase4_eval.py \
    --adapter-epoch "$EPOCH" --shard 1-of-2 "${EXTRA_FLAGS[@]}" \
    > "$LOG_DIR/phase4_ep${EPOCH}_gpu1.log" 2>&1 &
GPU1_PID=$!

echo "    GPU 0 PID: $GPU0_PID; GPU 1 PID: $GPU1_PID"
RC0=0; RC1=0
wait "$GPU0_PID" || RC0=$?
wait "$GPU1_PID" || RC1=$?
if [ "$RC0" -ne 0 ] || [ "$RC1" -ne 0 ]; then
    echo "FATAL: Phase 4 ep=${EPOCH} shard failed (gpu0 exit=$RC0, gpu1 exit=$RC1). See $LOG_DIR/phase4_ep${EPOCH}_gpu{0,1}.log" >&2
    exit 1
fi

echo "=== Phase 4 ep=${EPOCH} shards complete; running merger ==="
uv run python scripts/i462_phase4_merge.py --adapter-epoch "$EPOCH"

# Disk hygiene: with 4 ckpts × 16 conds = 64 adapters at ~150 MB each
# (~10 GB), the local adapter cache grows linearly per level. Wipe the
# per-level cache after the merger writes G_logprob_matrix_ep{N}.json so
# the next level starts clean. Adapters remain on HF (re-downloaded
# per level).
if [ -d /workspace/adapters/i462 ]; then
    echo "    Cleaning /workspace/adapters/i462 ($(du -sh /workspace/adapters/i462 | cut -f1)) ..."
    rm -rf /workspace/adapters/i462
fi

echo "=== Phase 4 ep=${EPOCH} done $(date -Iseconds) ==="
