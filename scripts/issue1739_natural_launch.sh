#!/usr/bin/env bash
# Same detach/pid contract as issue1739_uladder_launch.sh; never overwrite a live run.
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "usage: $0 <unique-run-name> <natural_run phase and arguments...>" >&2
  exit 2
fi
RUN_NAME="$1"
shift
case "$RUN_NAME" in
  *[!a-z0-9-]* | "") echo "invalid run name" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${EPM_LOG_DIR:-/workspace/logs}"
LOG="$LOG_DIR/issue-1739-natural-$RUN_NAME.log"
PID_FILE="$LOG_DIR/issue-1739-natural-$RUN_NAME.pid"
PY="$REPO_ROOT/.venv/bin/python"
mkdir -p "$LOG_DIR"
if [ -e "$LOG" ] || [ -e "$PID_FILE" ]; then
  echo "run name already used; choose a fresh launch name (existing logs preserved)" >&2
  exit 3
fi
if [ ! -x "$PY" ]; then
  echo "missing project interpreter: $PY" >&2
  exit 4
fi
export UV_NO_SYNC=1 PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export EPS_NATURAL_LAUNCH_LOG="$LOG"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2
cd "$REPO_ROOT"
nohup setsid "$PY" scripts/issue1739_natural_run.py "$@" </dev/null >"$LOG" 2>&1 &
worker_pid=$!
printf '%s\n' "$worker_pid" >"$PID_FILE"
printf 'pid=%s\npid_file=%s\nlog=%s\n' "$worker_pid" "$PID_FILE" "$LOG"
