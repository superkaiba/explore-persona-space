#!/usr/bin/env bash
# Issue #923 GPU phase — N-way context-sharded gen+capture fan-out (plan §4.3).
#
# Fresh dispatcher (the #658 8-GPU script is BRANCH-only — never sourced).
# Shards issue923_capture.py across EVERY visible GPU (wave size = detected GPU
# count, never a hardcoded constant), with CUDA_VISIBLE_DEVICES pinned in the
# LAUNCHER env per shard (the import-time-cuInit clobber gotcha). After the
# shards join: ONE upload invocation (single upload_folder commits + verify +
# UPLOAD_COMPLETE.json sentinel for the Phase-3 HF-poll join), then the full-H
# dual-ridge spot-check on GPU (EPM_FIT_DEVICE=cuda).
#
# Usage (dispatched via dispatch_issue.py --intent ft-7b --workload-cmd
# "bash scripts/issue923_gpu_phase.sh"):
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue_923}"
mkdir -p "$LOG_DIR"

NGPU=$(nvidia-smi --list-gpus | wc -l)
if [ "$NGPU" -lt 1 ]; then
  echo "[issue923_gpu_phase] FATAL: no visible GPUs" >&2
  exit 2
fi
echo "[issue923_gpu_phase] fan-out across $NGPU GPUs"

PIDS=()
for k in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="$k" \
  PYTHONUNBUFFERED=1 \
    nohup uv run python scripts/issue923_capture.py \
      --shard "$k/$NGPU" \
      --phases gen,tf,ffull,partials \
      "$@" \
      > "$LOG_DIR/capture_shard${k}.log" 2>&1 < /dev/null &
  PIDS+=($!)
  echo "[issue923_gpu_phase] shard $k/$NGPU pid=${PIDS[-1]} log=$LOG_DIR/capture_shard${k}.log"
done

RC=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[issue923_gpu_phase] shard $i FAILED (see $LOG_DIR/capture_shard${i}.log)" >&2
    tail -n 40 "$LOG_DIR/capture_shard${i}.log" >&2 || true
    RC=1
  fi
done
if [ "$RC" -ne 0 ]; then
  echo "[phase=failed]"
  exit "$RC"
fi

echo "[issue923_gpu_phase] shards complete; uploading (single invocation)"
uv run python scripts/issue923_capture.py --shard "0/$NGPU" --phases upload "$@" \
  2>&1 | tee "$LOG_DIR/upload.log"

echo "[issue923_gpu_phase] full-H dual-ridge spot-check (EPM_FIT_DEVICE=cuda)"
CUDA_VISIBLE_DEVICES=0 EPM_FIT_DEVICE=cuda \
  uv run python scripts/issue923_fit_decomposition.py --fullh-spotcheck \
  2>&1 | tee "$LOG_DIR/fullh_spotcheck.log"

echo "[phase=done]"
