#!/usr/bin/env bash
# Issue #2676 — block until an operator-staged marker file exists.
#
# The follow-up rerun reuses checkpoints the orchestrator copies onto the pod
# after it boots. The workload runs the smoke, then this script, then the real
# job list, so the job list never starts before staging finishes.
#
# usage: bash scripts/issue2676_wait_for_file.sh <path> <timeout_seconds>
# Exits 0 once <path> exists; exits 9 (loud) if it never appears.
set -euo pipefail
path="${1:?usage: issue2676_wait_for_file.sh <path> <timeout_seconds>}"
timeout_s="${2:?usage: issue2676_wait_for_file.sh <path> <timeout_seconds>}"
start=$(date +%s)
echo "[phase=wait_stage] waiting for $path (timeout ${timeout_s}s)"
while [ ! -f "$path" ]; do
    elapsed=$(( $(date +%s) - start ))
    if [ "$elapsed" -ge "$timeout_s" ]; then
        echo "[phase=wait_stage] FATAL: $path not staged after ${elapsed}s" >&2
        exit 9
    fi
    echo "[wait_stage] not yet staged, elapsed=${elapsed}s"
    sleep 30
done
echo "[phase=wait_stage] staged marker present: $path"
