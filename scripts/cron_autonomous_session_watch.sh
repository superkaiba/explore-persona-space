#!/bin/bash
# Crash-recovery watch for autonomous (`--auto`) issue sessions — invoked from
# the system crontab (every ~10 min). Re-spawns an autonomous /issue session
# whose driver process has died (crash / OOM / VM reboot), which the in-session
# /loop + durable=false cron cannot recover on their own. Mirrors
# cron_worktree_audit.sh / cron_pod_audit.sh.
#
# Safety lives in scripts/autonomous_session_watch.py: single-flight flock,
# a 2-consecutive-miss guard before any respawn, worktree-cwd liveness
# cross-check, and respawn ONLY for active-drive statuses (never for parked /
# awaiting_promotion tasks). See that file's docstring for the full rule.
#
# Output: logs/autonomous_session_watch/YYYY-MM-DD.log (one file per day).

set -uo pipefail

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="$PROJECT_DIR/logs/autonomous_session_watch"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

{
    echo "=== $(date -Iseconds) autonomous_session_watch start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/autonomous_session_watch.py
    rc=$?
    echo "=== $(date -Iseconds) autonomous_session_watch exit=$rc ==="
} >> "$LOG_FILE" 2>&1

# Exit 0 regardless — the log file is the audit trail; we don't want cron email
# on every routine "all sessions alive" pass or transient respawn.
exit 0
