#!/usr/bin/env bash
# Phase 3 — 2-GPU parallel cross-eval dispatcher.
#
# Issue #406 plan v9 §4 Phase 3.
#
# Splits the 16 active outer-i conditions across 2 GPUs by i_idx % 2
# (N=16 after 2026-05-31 C2-C5 scope drop). Each shard owns one vLLM
# process (enable_lora=True, one LoRA swap per outer-i, inner-j x 50
# q_test batched into one llm.generate() call per (i, j) pair). After
# both shards finish, the merger combines the per-shard roll-ups into
# G_matrix.json.
#
# Usage:
#     bash scripts/i406_phase3_dispatch.sh
#     bash scripts/i406_phase3_dispatch.sh --resume

set -euo pipefail
EXTRA_FLAGS="${*:-}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

LOG_DIR=logs/issue_406
mkdir -p "$LOG_DIR"

echo "=== Phase 3 dispatcher starting at $(date -Iseconds) ==="
echo "    Extra flags forwarded to each shard: ${EXTRA_FLAGS:-<none>}"

CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/i406_phase3_cross_eval.py \
    --shard 0-of-2 ${EXTRA_FLAGS} \
    > "$LOG_DIR/phase3_gpu0.log" 2>&1 &
GPU0_PID=$!

CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts/i406_phase3_cross_eval.py \
    --shard 1-of-2 ${EXTRA_FLAGS} \
    > "$LOG_DIR/phase3_gpu1.log" 2>&1 &
GPU1_PID=$!

echo "    GPU 0 PID: $GPU0_PID; GPU 1 PID: $GPU1_PID"
# Capture each shard's exit code separately (see Phase 1 dispatcher note):
# plain `wait $P0 $P1` returns only the last pid's status and would mask a
# crashed shard, letting the merger run on partial/missing per-shard output.
RC0=0; RC1=0
wait "$GPU0_PID" || RC0=$?
wait "$GPU1_PID" || RC1=$?
if [ "$RC0" -ne 0 ] || [ "$RC1" -ne 0 ]; then
    echo "FATAL: Phase 3 shard failed (gpu0 exit=$RC0, gpu1 exit=$RC1). See $LOG_DIR/phase3_gpu{0,1}.log" >&2
    exit 1
fi

echo "=== Phase 3 shards complete; running merger ==="
uv run python scripts/i406_phase3_merge_g_matrix.py

echo "=== Phase 3 done at $(date -Iseconds) ==="
