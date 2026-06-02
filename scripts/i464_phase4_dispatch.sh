#!/usr/bin/env bash
# Phase 4 — cross-eval marker log-prob sharded across 2 GPUs.
#
# Issue #464 plan v2 §4.1 Phase 4 + §9.1. Splits the 9 cells across 2
# GPUs (round-robin: 5+4). Each shard runs its own vLLM engine with LoRA
# hot-swap. Per-cell atomic writes for crash safety + --resume.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

EXTRA_FLAGS="${*:-}"
LOG_DIR=logs/issue_464
mkdir -p "$LOG_DIR"

echo "=== Phase 4 dispatcher starting $(date -Iseconds) ==="
echo "    Extra flags: ${EXTRA_FLAGS:-<none>}"

CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/i464_phase4_eval.py \
    --shard 0-of-2 ${EXTRA_FLAGS} \
    > "$LOG_DIR/phase4_gpu0.log" 2>&1 &
GPU0_PID=$!

CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts/i464_phase4_eval.py \
    --shard 1-of-2 ${EXTRA_FLAGS} \
    > "$LOG_DIR/phase4_gpu1.log" 2>&1 &
GPU1_PID=$!

echo "    GPU 0 PID: $GPU0_PID; GPU 1 PID: $GPU1_PID"
RC0=0; RC1=0
wait "$GPU0_PID" || RC0=$?
wait "$GPU1_PID" || RC1=$?
if [ "$RC0" -ne 0 ] || [ "$RC1" -ne 0 ]; then
    sentinel="/workspace/logs/issue-464-phase4-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 464,
    "phase": "phase4_eval",
    "failure_class": "code",
    "shard_exits": {"gpu0": $RC0, "gpu1": $RC1},
    "reason": "phase4 eval shard exited non-zero",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
EOF
    echo "FATAL: Phase 4 shard failed (gpu0=$RC0, gpu1=$RC1)." >&2
    exit 1
fi
echo "=== Phase 4 done $(date -Iseconds) ==="
