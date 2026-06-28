#!/bin/bash
# Deterministic failure-lesson consolidation — invoked from system crontab (task #711).
# Extracted from the nightly /daily LLM run so the dedupe/promote/prune pass over
# epm:failure-lesson markers no longer depends on a flaky 44K-token LLM run.
#
# Runs scripts/consolidate_lessons.py --apply over a rolling 7-day window:
#   (a) dedupe same-window lessons against the owning agent's memory;
#   (b) promote recurring lessons (>=2 distinct tasks) into .claude/rules/gotchas.md;
#   (c) prune over-eager generalizes: yes memory entries.
# Idempotent (a 2nd run with no new markers makes no commit) and fails loud on
# corrupting-write / git / unreadable-target conditions.
#
# Output lives at logs/lesson_consolidate/YYYY-MM-DD.log (the consolidator's own
# counts line + this wrapper's start/exit pointers; one file per day, date-stamped
# so no rotation is needed).

set -uo pipefail

# cron's minimal PATH lacks ~/.local/bin, so a bare `uv` exit-127s silently
# (the `exit 0` below hides it). Put uv on PATH; fail LOUD if still missing.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); cannot run lesson consolidation" >&2
    exit 1
fi

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="${EPS_LESSON_CONSOLIDATE_LOG_DIR:-$PROJECT_DIR/logs/lesson_consolidate}"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

# One pointer line per day into the crontab redirect file: everything below
# runs inside a block redirected to $LOG_FILE, so without this the redirect
# file stays empty forever and reads as "the cron never ran" (task #580
# item-3 diagnosis; mirrors cron_pod_audit.sh).
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

{
    echo "=== $(date -Iseconds) lesson_consolidate start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/consolidate_lessons.py --apply --window-days 7
    rc=$?
    echo "=== $(date -Iseconds) lesson_consolidate exit=$rc ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) lesson_consolidate: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 regardless — the log file is the audit trail, no cron email per routine pass.
exit 0
