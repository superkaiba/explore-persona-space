#!/bin/bash
# Periodic LLM summary of every live EPS Happy session (every 5 min).
# See scripts/session_summarize.py: resolves each live session's Claude Code
# transcript, reads the tail, asks Haiku to summarize what the session is
# DOING right now, and writes ~/.eps-autonomous/session_progress.json.
# Read by the dashboard + `spawn_session.py list` (PROGRESS column).
#
# Output: logs/session_summarize/YYYY-MM-DD.log (one file per day).
# Mirrors cron_autonomous_session_watch.sh / cron_worktree_audit.sh.
# Fails loud (stderr FATAL + exit 1) when the log dir cannot be created or the
# daily log file is not appendable (task #2386; ports the #2196 pattern).
# TEST-ONLY env seam: EPS_SESSION_SUMMARIZE_LOG_DIR (log dir).

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
LOG_DIR="${EPS_SESSION_SUMMARIZE_LOG_DIR:-$PROJECT_DIR/logs/session_summarize}"
LOG_FILE="$LOG_DIR/$DATE.log"

# Fail-loud helper for wrapper-infrastructure failures (task #2386, ports the
# #2196 pattern): an unchecked failure here silently skips the whole pass
# below (the brace-group redirect fails and the group never runs) while the
# wrapper still exits 0. stderr lands in the crontab redirect file where one
# exists; cron mail is structurally dead on this VM (no MTA).
fatal() {
    echo "$(date -Iseconds) FATAL: $1" >&2
    exit 1
}

mkdir -p "$LOG_DIR" \
    || fatal "cannot create log dir (LOG_DIR=$LOG_DIR); session summarize pass NOT run"

# mkdir -p succeeds on an existing dir regardless of writability, so probe the
# actual append open the brace group below will attempt. This wrapper has no
# FIRST_RUN_OF_DAY read to sit after (no daily pointer line), so the probe goes
# directly after the guarded mkdir; it creates $LOG_FILE when absent.
: >> "$LOG_FILE" 2>/dev/null \
    || fatal "daily log file not appendable ($LOG_FILE); session summarize pass NOT run"

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
