#!/bin/bash
# Daily stale-worktree sweep — invoked from system crontab.
# Safety net for the /issue Step 10d worktree removal that does not always
# fire, leaving auto-generated worktrees (issue-<N>, agent-<hex>, wf_<id>)
# under .claude/worktrees/ to pile up (102 worktrees / 161 GB had
# accumulated by 2026-05-28). Mirrors cron_pod_audit.sh.
#
# Policy (see scripts/worktree_audit.py for the full rule): an auto-generated
# worktree is removed only when it is provably idle — not held by a live
# process, not a non-terminal issue status, older than the 6h grace window
# (tightened to 1h when the filesystem holding the worktrees is >=90% full —
# disk-pressure mode, threshold via EPM_WORKTREE_DISK_PRESSURE_PCT), and with
# no uncommitted tracked changes. Human-named worktrees are never touched
# (issue-<N>-<suffix> follow-up worktrees ARE in scope as of 2026-06-12,
# mapped to issue N for the status lookup).
# For done-and-merged (completed/archived/awaiting_promotion) issue
# worktrees, --apply additionally
# remediates two false-keep classes (2026-06-10 disk-full incident): kills
# orphaned codex app-server holder pids (exact-pid, cmdline re-verified;
# never when a real holder is present) and rescue-copies allowlisted
# runtime-noise dirt (agent memories, pods.conf, pods_ephemeral.json) to
# .claude/cache/worktree-rescue-<date>/ BEFORE removal. Dry-run only
# classifies — it never kills or rescues.
# --apply additionally reaps the .venv of kept worktrees idle >=7d (2d under
# pressure) with no live holder (#912); regenerable via `uv run`;
# EPM_WORKTREE_VENV_REAP=0 disables.
#
# Output lives at logs/worktree_audit/YYYY-MM-DD.log (one file per day).

set -uo pipefail

# cron's minimal PATH lacks ~/.local/bin, so a bare `uv` exit-127s silently
# (the `exit 0` below hides it). Put uv on PATH; fail LOUD if still missing.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); cannot run worktree audit" >&2
    exit 1
fi

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="${EPM_WORKTREE_AUDIT_LOG_DIR:-$PROJECT_DIR/logs/worktree_audit}"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

# One pointer line per day into the crontab redirect file: everything below
# runs inside a block redirected to $LOG_FILE, so without this the redirect
# file stays empty forever and reads as "the audit never ran" (task #580
# item-3 diagnosis, 2026-06-12; mirrors cron_autonomous_session_watch.sh).
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

{
    echo "=== $(date -Iseconds) worktree_audit start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/worktree_audit.py --apply
    rc=$?
    echo "=== $(date -Iseconds) worktree_audit exit=$rc ==="
    # #1430: terminal-task duplicate-dir husk reap (subset-verified,
    # escalate-never-delete; kill switch honored inside the library via
    # EPM_SKIP_HUSK_REAP=1).
    uv run python scripts/task.py reap-husks --apply
    echo "=== $(date -Iseconds) husk_reap exit=$? ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) worktree_audit: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 even if the audit returned 2 — we don't want cron emails on every
# "found and removed stale worktree" event. The log file is the audit trail.
exit 0
