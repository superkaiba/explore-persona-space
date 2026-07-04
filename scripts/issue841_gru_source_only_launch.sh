#!/usr/bin/env bash
# Issue #841 follow-up (gru-source-only) setsid launcher — detaches the phase driver so it
# outlives the SSH/launch shell (the #883 launcher shape). Writes a PID file + a phase log
# the poller / experimenter tails; exits immediately.
#
# Usage (on the pod / GCE, after sync + preflight):
#   bash scripts/issue841_gru_source_only_launch.sh
#   EPM_I841GSO_SMOKE=1 bash scripts/issue841_gru_source_only_launch.sh   # unified smoke

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

LOGS_DIR="${EPM_LOGS_DIR:-/workspace/logs}"
mkdir -p "$LOGS_DIR"
LOG="$LOGS_DIR/issue-841-gru-source-only-$(date +%Y%m%d-%H%M%S).log"
PIDFILE="$LOGS_DIR/issue-841-gru-source-only.pid"

# setsid + nohup so the driver survives the launch shell exiting; </dev/null so the SSH
# call does not hang on the backgrounded child (gotchas.md).
PHASE_PID=$(bash -c "setsid nohup bash scripts/issue841_gru_source_only_dispatch.sh \
  < /dev/null >> '$LOG' 2>&1 & echo \$!")
echo "$PHASE_PID" > "$PIDFILE"
echo "launched issue-841 gru-source-only driver pid=$PHASE_PID log=$LOG pidfile=$PIDFILE"
echo "tail -f $LOG   # [phase=...] breadcrumbs; terminal [phase=done] on success"
