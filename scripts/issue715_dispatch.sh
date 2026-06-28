#!/usr/bin/env bash
# Issue #715 — thin launcher for the pod-side / GCP-lane dispatcher.
#
# Wraps scripts/issue715_dispatch.py so the GCP --workload-cmd lane and the
# RunPod nohup launch share ONE entrypoint. Redirects the workload's stdout to a
# LOG FILE (never streamed through the GCE metadata-script-runner stdout pipe —
# the bufio.Scanner "token too long" zombie, gotchas.md) and exits non-zero on
# dispatcher failure so the EXIT-trap teardown fires.
#
# Usage (GCP --workload-cmd): REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue715_dispatch.sh --phase phase1
# Usage (RunPod nohup): nohup bash scripts/issue715_dispatch.sh --phase phase1 > log 2>&1 < /dev/null &
set -euo pipefail

# REPO_ROOT resolves transparently on GCP (startup script exports WORKLOAD_ROOT);
# the GCE workload_cmd branch already exports REPO_ROOT (#641), so the default is
# the belt-and-suspenders fallback.
REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Credentials at entry: uv run does NOT auto-load .env; the python dispatcher
# also calls load_dotenv(), but assert here so a missing-key launch fails loud.
set -a && source .env 2>/dev/null && set +a || true

LOG_DIR="${EPS_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="$REPO_ROOT/eval_results/issue_715/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/issue715-dispatch-$(date +%s).log"

echo "[issue715] launching dispatcher: args=$* log=$LOG" | tee -a "$LOG"
# TQDM_DISABLE so vLLM/tqdm progress bars never produce giant newline-free lines
# (the metadata-runner bufio.Scanner zombie + the #613 ZeroDivisionError class).
TQDM_DISABLE=1 uv run python scripts/issue715_dispatch.py "$@" >>"$LOG" 2>&1
rc=$?
echo "[issue715] dispatcher exited rc=$rc" | tee -a "$LOG"
exit $rc
