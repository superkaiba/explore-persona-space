#!/usr/bin/env bash
# Phase 4 — cross-eval marker log-prob (DV: delta_g) sharded across 2 GPUs.
#
# Issue #460 plan v3 §4.6 + §9.1. Splits 16 outer-i conds across 2 GPUs;
# each shard runs its own vLLM engine with LoRA hot-swap. After both shards
# finish, the merger combines per-shard roll-ups + per-cell atomic writes
# into the single G_logprob_matrix.json.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

EXTRA_FLAGS="${*:-}"
LOG_DIR=logs/issue_460
mkdir -p "$LOG_DIR"

echo "=== Phase 4 dispatcher starting $(date -Iseconds) ==="
echo "    Extra flags: ${EXTRA_FLAGS:-<none>}"

CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/i460_phase4_eval.py \
    --shard 0-of-2 ${EXTRA_FLAGS} \
    > "$LOG_DIR/phase4_gpu0.log" 2>&1 &
GPU0_PID=$!

CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts/i460_phase4_eval.py \
    --shard 1-of-2 ${EXTRA_FLAGS} \
    > "$LOG_DIR/phase4_gpu1.log" 2>&1 &
GPU1_PID=$!

echo "    GPU 0 PID: $GPU0_PID; GPU 1 PID: $GPU1_PID"
RC0=0; RC1=0
wait "$GPU0_PID" || RC0=$?
wait "$GPU1_PID" || RC1=$?
if [ "$RC0" -ne 0 ] || [ "$RC1" -ne 0 ]; then
    echo "FATAL: Phase 4 shard failed (gpu0 exit=$RC0, gpu1 exit=$RC1). See $LOG_DIR/phase4_gpu{0,1}.log" >&2
    exit 1
fi

echo "=== Phase 4 shards complete; running merger ==="
uv run python scripts/i460_phase4_merge.py
echo "=== Phase 4 done $(date -Iseconds) ==="
