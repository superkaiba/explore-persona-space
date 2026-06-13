#!/bin/bash
# Periodic LLM summary of every live EPS Happy session (every 5 min).
# See scripts/session_summarize.py: resolves each live session's Claude Code
# transcript, reads the tail, asks Haiku to summarize what the session is
# DOING right now, and writes ~/.eps-autonomous/session_progress.json.
# Read by the dashboard + `spawn_session.py list` (PROGRESS column).
#
# Output: logs/session_summarize/YYYY-MM-DD.log (one file per day).
# Mirrors cron_autonomous_session_watch.sh / cron_worktree_audit.sh.

set -uo pipefail

# cron's minimal PATH won't have ~/.local/bin where uv lives; surface a LOUD
# failure if the binary moved instead of silently exit-127ing under cron.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); cannot summarize" >&2
    exit 1
fi

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="$PROJECT_DIR/logs/session_summarize"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

{
    echo "=== $(date -Iseconds) session_summarize start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/session_summarize.py
    rc=$?
    echo "=== $(date -Iseconds) session_summarize exit=$rc ==="
    # Task-progress snapshot (task #587): this cron is the ONLY writer of
    # ~/.eps-autonomous/task_progress.json (dashboard + title-suffix reader).
    uv run python scripts/task_progress.py snapshot
    echo "=== $(date -Iseconds) task_progress snapshot exit=$? ==="
} >> "$LOG_FILE" 2>&1

# Exit 0 regardless — the log file is the audit trail; we don't want cron mail
# for routine ticks (the dashboard surfaces failures by showing stale entries).
exit 0
