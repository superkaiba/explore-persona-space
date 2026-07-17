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

# #1466: fleet tmux socket dir (cron env carries no TMUX_TMPDIR; this
# wrapper drives tmux_window_titles.py against the fleet server).
. "$PROJECT_DIR/scripts/eps_tmux_env.sh"

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
    # Rename each live tmux WINDOW to a short summary of what it is doing, so
    # `tmux ls` / the switcher are browsable for resume. Reuses the cache
    # written just above for EPS sessions; Haiku-summarizes the rest (idle-
    # skipped). Session names are left unchanged. See tmux_window_titles.py.
    uv run python scripts/tmux_window_titles.py apply
    echo "=== $(date -Iseconds) tmux_window_titles exit=$? ==="
} >> "$LOG_FILE" 2>&1

# Exit 0 regardless — the log file is the audit trail; we don't want cron mail
# for routine ticks (the dashboard surfaces failures by showing stale entries).
exit 0
