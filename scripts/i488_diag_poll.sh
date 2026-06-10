#!/bin/bash
# Poll pod-488 for diagnostic train/measure progress. Single tick.
# Returns JSON to stdout: {"phase": "train"|"measure"|"done"|"failed", ...}
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Sourcing pod info from synced pods.conf at repo root.
POD="pod-488"
PROD_LOG="/workspace/logs/i488_diag_train.log"
MEAS_LOG="/workspace/logs/i488_diag_measure.log"
ADAPTER_BASE="/workspace/adapters/i488_diag"
PROBES_PATH="/workspace/explore-persona-space/eval_results/issue_488/diagnostic_separation/probes.json"
MEAS_PID_FILE="/workspace/logs/i488_diag_measure.pid"

ssh_run() {
  uv run python scripts/pod.py ssh "$POD" -- "$@" 2>&1 || true
}

# Single SSH call collecting all state.
read -r -d '' STATE_CMD <<'BASH' || true
echo "===TRAIN_PIDS==="
pgrep -af i488_diagnostic_train.py | head -5
echo "===MEAS_PIDS==="
pgrep -af i488_diagnostic_measure.py | head -5
echo "===ADAPTERS==="
ls /workspace/adapters/i488_diag/ 2>/dev/null
echo "===PROBES==="
ls -la /workspace/explore-persona-space/eval_results/issue_488/diagnostic_separation/probes.json 2>/dev/null || echo "no probes yet"
echo "===TRAIN_LAST==="
tail -5 /workspace/logs/i488_diag_train.log 2>/dev/null | tail -3
echo "===MEAS_LAST==="
tail -5 /workspace/logs/i488_diag_measure.log 2>/dev/null | tail -3 || true
BASH

ssh_run "$STATE_CMD"
