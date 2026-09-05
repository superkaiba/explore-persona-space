#!/usr/bin/env bash
# Detached launcher for issue #1739 U-ladder pod runs.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <run-name> [issue1739_uladder_run.py args...]" >&2
  exit 2
fi

RUN_NAME="$1"
shift
case "$RUN_NAME" in
  *[!a-z0-9-]* | "")
    echo "run-name must contain only lowercase letters, digits, and hyphens" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${EPM_LOG_DIR:-/workspace/logs}"
LOG="$LOG_DIR/issue-1739-uladder-$RUN_NAME.log"
PID_FILE="$LOG_DIR/issue-1739-uladder-$RUN_NAME.pid"
ARCHIVE="$LOG_DIR/archive/issue-1739-uladder-$RUN_NAME"
PY="$REPO_ROOT/.venv/bin/python"

mkdir -p "$LOG_DIR"
if [ -e "$LOG" ] || [ -e "$PID_FILE" ]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$ARCHIVE"
  [ ! -e "$LOG" ] || mv "$LOG" "$ARCHIVE/$stamp.log"
  [ ! -e "$PID_FILE" ] || mv "$PID_FILE" "$ARCHIVE/$stamp.pid"
fi
if [ ! -x "$PY" ]; then
  echo "missing executable environment python: $PY" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export UV_NO_SYNC=1
cd "$REPO_ROOT"
nohup setsid "$PY" scripts/issue1739_uladder_run.py "$@" \
  </dev/null >"$LOG" 2>&1 &
worker_pid=$!
pid_tmp="$PID_FILE.tmp.$$"
printf '%s\n' "$worker_pid" >"$pid_tmp"
mv "$pid_tmp" "$PID_FILE"
printf 'pid=%s\npid_file=%s\nlog=%s\n' "$worker_pid" "$PID_FILE" "$LOG"
