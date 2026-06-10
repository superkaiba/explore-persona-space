#!/bin/bash
# VM-health + crash-recovery + pod-safety + stalled-detector + orphan-sweep
# watch for issue sessions — invoked from the system crontab (every ~10 min).
# Six passes, in order (see scripts/autonomous_session_watch.py's module
# docstring for the full rules):
#   1. VM disk-headroom: alert when free space on the VM root filesystem runs
#      low (~20 GiB); below ~8 GiB also run safe fail-soft reclaims. A full /
#      silently kills every foreground Bash spawn in orchestrator sessions
#      (task #552).
#   2. Crash-recovery: respawn a recoverable autonomous (`--auto`) /issue
#      session whose driver process has died (crash / OOM / VM reboot), which
#      the in-session /loop + durable=false cron cannot recover on their own.
#   3. Pod-safety: AUTO-STOP (NOT terminate) a RUNNING managed pod-<N> /
#      legacy epm-issue-<N> pod whose task is already DONE; ALERT (no stop)
#      on a pod-active task with no marker progress for hours — bounding GPU
#      burn instead of letting an escaped pod run to the 7-day TTL.
#   4. Stalled-detector: detect a live-but-frozen session (self-report AND
#      latest progress marker both stale >45 min) and auto-respawn it
#      (bounded per episode); alert-only for manual sessions or when the
#      Happy daemon is unreachable.
#   5. Orphan sweep: registration-INDEPENDENT cross-check — any ACTIVE-status
#      task with NO live registered session AND no real progress marker for
#      ~90 min (EPM_ORPHAN_STALENESS_MIN) is auto-respawned (capped at 2
#      attempts/task/day, EPM_ORPHAN_RESPAWNS_PER_DAY); alert-only for
#      manual-registered tasks. Closes the #472/#518 blind spot (2026-06-10):
#      a task revived by a same-issue follow-up with no registration, or one
#      whose registered driver died while a zombie generation masked it.
#   6. GC: reap per-issue watcher state files for completed/archived tasks.
# Mirrors cron_worktree_audit.sh / cron_pod_audit.sh.
#
# Safety lives in scripts/autonomous_session_watch.py: single-flight flock, a
# 2-consecutive-miss guard before any respawn OR pod-stop, worktree-cwd liveness
# cross-check, respawn ONLY for active-drive statuses (never for parked /
# awaiting_promotion tasks), pod-stop keyed on TASK STATUS proving the run is
# done (never on session liveness), and a daemon-reachability guard that skips
# the respawn + stalled-respawn arms (the passes that reason about session
# liveness) during an outage — the pod-safety, disk, and GC passes run
# regardless. See that file's docstring for the full rule.
#
# Output: logs/autonomous_session_watch/YYYY-MM-DD.log (one file per day).

set -uo pipefail

# cron runs with a minimal PATH (no ~/.local/bin), so a bare `uv` is "command
# not found" and the script silently exit-127s (the `exit 0` below hides it).
# Put uv on PATH and fail LOUD if it is still missing, so a PATH regression
# surfaces (cron mail) instead of silently disabling crash recovery.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); cannot run watcher" >&2
    exit 1
fi

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

# SESSION-RECONCILE PASS IS ALERT-ONLY PERMANENTLY (user decision, 2026-06-10):
# do NOT export EPM_SESSION_RECONCILE_AUTOSTOP here or anywhere else. The
# watcher may only ALERT on idle sessions of completed/archived tasks;
# stopping them stays a manual user action.
