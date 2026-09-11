#!/usr/bin/env bash
set -euo pipefail

# The GCP startup contract waits for fresh /workspace/logs/*.pid files after
# the original blocking workload exits. Register this continuation immediately,
# then its driver waits for the exact original PID before using the GPU.
SOURCE_SHA="$1"
PARENT_PID="$2"
LOSODIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LOSODIR"
export PYTHONPATH="$LOSODIR/src"
export PYTHONUNBUFFERED=1
export EPS_SENTINEL_PATH=/workspace/issue2054_k5/leave_one_setting_out_v1/.completion-sentinel.json
printf '%s\n' "$$" > /workspace/logs/issue-2054-k5-loso.pid.tmp
mv /workspace/logs/issue-2054-k5-loso.pid.tmp /workspace/logs/issue-2054-k5-loso.pid
exec uv run --no-sync python scripts/issue2054_k5_loso.py \
  --parent-root /workspace/issue2054_k5/production_v1 \
  --out-root /workspace/issue2054_k5/leave_one_setting_out_v1 \
  --source-sha "$SOURCE_SHA" --wait-parent-pid "$PARENT_PID" --device cuda
