#!/bin/bash
# VM root-disk guard — invoked from the USER crontab (thomasjiralerspong).
# Reads df for / and, when usage exceeds the threshold (default 85%, env
# EPS_VM_DISK_THRESHOLD), runs five tiers of strictly-safe cleanup:
#   (a) uv cache prune (never --force; skips gracefully if the lock is held);
#   (b) TERMINAL issues' data/issue_*/hf_dl + g*_dl caches (+ #911
#       non-canonical /tmp + data/ caches); store/ + eval_results/ NEVER
#       touched; task state read-only; active issues escalate-only;
#   (d) age-gated reap of the VM's pod-style /workspace/.cache/huggingface
#       hub cache (repos unused >= 14 d; pod-guarded; #911);
#   (e) HOME HF hub cache ~/.cache/huggingface/hub (#1376 + #1377): per-repo
#       attribution + >40 GB escalation always; reaps stale unref'd
#       non-newest revisions (>= 7 d; newest + every ref'd revision kept)
#       + wholly-stale repos via delete_revisions;
#   (c) logs/**/*.log + /tmp/*.log older than N days (env
#       EPS_VM_DISK_LOG_MAX_AGE_DAYS, default 14).
# After cleanup, if / is still over threshold it prints a loud WARNING and
# (fail-soft) phone-pushes via my-goat's telegram_push.sh for manual triage.
#
# The guard root disk hygiene was added after / hit 100% full on 2026-06-25
# (one finished experiment held 97 GB of re-downloadable hf_dl cache). The
# per-experiment cleanup is also wired into /issue Step 8 (post-upload-PASS);
# this cron is the fleet-wide backstop for caches that escape that path.
#
# See scripts/vm_disk_guard.py for the full tier policy. Output lives at
# logs/vm_disk_guard/YYYY-MM-DD.log (one file per day).

set -uo pipefail

# cron's minimal PATH lacks ~/.local/bin, so a bare `uv` exit-127s silently
# (the `exit 0` below would hide it). Put uv on PATH; fail LOUD if still missing.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); cannot run vm_disk_guard" >&2
    exit 1
fi

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="${EPS_VM_DISK_GUARD_LOG_DIR:-$PROJECT_DIR/logs/vm_disk_guard}"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

# One pointer line per day into the crontab redirect file: everything below
# runs inside a block redirected to $LOG_FILE, so without this the redirect
# file stays empty forever and reads as "the guard never ran" (mirrors
# cron_worktree_audit.sh / cron_autonomous_session_watch.sh).
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

{
    echo "=== $(date -Iseconds) vm_disk_guard start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/vm_disk_guard.py --apply
    rc=$?
    echo "=== $(date -Iseconds) vm_disk_guard exit=$rc ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) vm_disk_guard: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 even when the guard returned 2 (still-over-threshold) — the loud
# WARNING line + the telegram push inside the guard are the alarm channel; we
# don't want a cron email on every pass. The log file is the audit trail.
exit 0
