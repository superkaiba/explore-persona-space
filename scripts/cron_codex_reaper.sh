#!/bin/bash
# Recurring Codex app-server daemon reaper — invoked from system crontab.
# Reaps daemons older than EPS_CODEX_REAPER_MAX_AGE_H (default 24h) and truncates
# ~/.codex/logs_2.sqlite's WAL. Mirrors cron_pod_audit.sh / cron_worktree_audit.sh.
#
# Policy:
#   - Codex ensemble-review spawns a persistent daemon trio per session
#     (node codex app-server, its codex-linux-x64 vendor binary,
#     app-server-broker.mjs serve) that does NOT exit after the companion task
#     completes. They accumulate over weeks, each holding ~/.codex/logs_2.sqlite
#     (WAL mode) open so SQLite can never checkpoint the WAL.
#   - Daemons older than the threshold are SIGTERM'd (SIGKILL survivors), then
#     the WAL is best-effort PRAGMA wal_checkpoint(TRUNCATE)'d. The active
#     review drivers (codex_task.py) and sub-threshold daemons are spared.
#
# Output lives at logs/codex_reaper/YYYY-MM-DD.log (one file per day, no
# rotation needed because of the date stamp).

set -uo pipefail

# cron's minimal PATH lacks ~/.local/bin, so a bare `uv` exit-127s silently
# (the `exit 0` below hides it). Put uv on PATH; fail LOUD if still missing.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); cannot run codex reaper" >&2
    exit 1
fi

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="${EPM_CODEX_REAPER_LOG_DIR:-$PROJECT_DIR/logs/codex_reaper}"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

# One pointer line per day into the crontab redirect file: everything below
# runs inside a block redirected to $LOG_FILE, so without this the redirect
# file stays empty forever and reads as "the reaper never ran" (task #580
# item-3 diagnosis; mirrors cron_pod_audit.sh / cron_autonomous_session_watch.sh).
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

{
    echo "=== $(date -Iseconds) codex_reaper start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/codex_daemon_reaper.py --apply
    rc=$?
    echo "=== $(date -Iseconds) codex_reaper exit=$rc ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) codex_reaper: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 even if the reaper returned 2 (something reaped) or 3 (ps read failed)
# — we don't want cron emails on every reap. The log file is the audit trail.
exit 0
