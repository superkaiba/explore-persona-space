#!/bin/bash
# Daily stale-worktree sweep — invoked from system crontab.
# Safety net for the /issue Step 10d worktree removal that does not always
# fire, leaving auto-generated worktrees (issue-<N>, agent-<hex>, wf_<id>)
# under .claude/worktrees/ to pile up (102 worktrees / 161 GB had
# accumulated by 2026-05-28). Mirrors cron_pod_audit.sh.
#
# Policy (see scripts/worktree_audit.py for the full rule): an auto-generated
# worktree is removed only when it is provably idle — not held by a live
# process, not a non-terminal issue status, older than the 6h grace window,
# and with no uncommitted tracked changes. Human-named worktrees are never
# touched.
#
# Output lives at logs/worktree_audit/YYYY-MM-DD.log (one file per day).

set -uo pipefail

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="$PROJECT_DIR/logs/worktree_audit"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

{
    echo "=== $(date -Iseconds) worktree_audit start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/worktree_audit.py --apply
    rc=$?
    echo "=== $(date -Iseconds) worktree_audit exit=$rc ==="
} >> "$LOG_FILE" 2>&1

# Exit 0 even if the audit returned 2 — we don't want cron emails on every
# "found and removed stale worktree" event. The log file is the audit trail.
exit 0
