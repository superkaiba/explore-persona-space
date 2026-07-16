#!/bin/bash
# VM-health + crash-recovery + pod-safety + stalled-detector + orphan-sweep
# + infra-drain + session-reconcile + campaign watch for issue/campaign
# sessions — invoked from the system crontab (every ~10 min). Twelve passes
# (the campaign pass runs right after item 2; only the highlights are
# summarized below — the zombie-wrapper + idle-unmapped session reapers live
# only in scripts/autonomous_session_watch.py's module docstring, which
# carries the full, authoritative rules):
#   1. VM disk-headroom: alert when free space on the VM root filesystem runs
#      low (~20 GiB) AND run the stale-worktree sweep (worktree_audit.py
#      --apply — the big-space remediation; 6h re-arm); below ~15 GiB (env
#      EPM_VM_DISK_CRITICAL_GIB) also run the safe fail-soft cache reclaims
#      (wandb artifact cache cleanup, uv cache prune, npm cache clean, HF hub
#      TTL eviction of revisions idle > EPM_VM_DISK_HF_TTL_DAYS, stale
#      /tmp/claude-* sweep), each logging its freed space in the marker note.
#      A full / silently kills every foreground Bash spawn in orchestrator
#      sessions (task #552; remediation-at-detection added after #587,
#      2026-06-11; wandb/HF cache tiers added 2026-06-12 after the reclaims
#      freed ~0 while 17.6 GB wandb + 41.5 GB HF sat reclaimable).
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
#      Happy daemon is unreachable (daemon-blocked remediations persist and
#      auto-execute on the next daemon-reachable tick).
#   5. Orphan sweep: registration-INDEPENDENT cross-check — any ACTIVE-status
#      task with NO live registered session AND no real progress marker for
#      ~90 min (EPM_ORPHAN_STALENESS_MIN) is auto-respawned (capped at 2
#      attempts/task/day, EPM_ORPHAN_RESPAWNS_PER_DAY); alert-only for
#      manual-registered tasks. Closes the #472/#518 blind spot (2026-06-10):
#      a task revived by a same-issue follow-up with no registration, or one
#      whose registered driver died while a zombie generation masked it.
#   6. Session-reconcile: AUTO-STOP (default since 2026-06-10; set
#      EPM_SESSION_RECONCILE_AUTOSTOP=0 for the alert-only fallback) live
#      Happy sessions whose task is parked/terminal (awaiting_promotion /
#      completed / archived) once ALL hold across >=2 checks: no follow-up
#      signal marker newer than the latest done-transition, every non-watcher
#      marker + self-report idle > ~2h (EPM_SESSION_RECONCILE_IDLE_S), no
#      RUNNING managed pod for the issue, no keep-running tag. Never touches
#      unmapped sessions (PM / chat), followups_running, or blocked tasks.
#   7. GC: reap per-issue watcher state files for completed/archived tasks.
#   8. Campaign pass (runs right after item 2; task #586): respawn a dead
#      /campaign session via spawn-campaign; epm:campaign-stalled alert +
#      bounded stop-then-respawn when neither the campaign nor any child has
#      posted a marker for EPM_CAMPAIGN_STALL_S (default 2h); one-time alert
#      when campaign-state.json shows GPU-hours committed > total; at
#      terminal status stop the live session FIRST, then reap the
#      campaign-<N>.json entry (deferred while the daemon is unreachable).
#   9. Infra-drain (runs between items 5 and 6; task #633): execute the PM
#      session's adjudicated dispatch queue
#      (~/.eps-autonomous/infra-drain-queue.json) — spawn-issue --auto the
#      oldest listed kind:infra/batch IDs still at proposed, into free slots
#      under the cap (default 3), skipping holds / existing registrations /
#      mis-kinded IDs, with a 1h-backoff + 3-attempts-per-PM-epoch budget
#      (EPM_INFRA_DRAIN_BACKOFF_S / EPM_INFRA_DRAIN_MAX_ATTEMPTS) so a
#      failing spawn never tight-loops. The PM is the ONLY ripeness judge;
#      a missing/invalid queue file is a logged no-op. Kill switch:
#      EPM_DISABLE_INFRA_DRAIN=1.
#  10. CPU/memory-pressure guard (task #849; daemon-independent, right after
#      the disk/happy-patch checks): escalate-only detection + attribution of
#      VM compute pressure (streaked load5/PSI, single-tick MemAvailable
#      floor) + silent earlyoom SIGTERM kill surfacing with pre-kill-snapshot
#      attribution_status — sidecar .claude/cache/cpu-guard-events.jsonl +
#      deduped push. WARN-ONLY (never kills/renices). Kill switch:
#      EPM_DISABLE_CPU_GUARD_PASS=1; --cpu-guard-only for a live smoke.
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
LOG_DIR="${EPM_WATCH_LOG_DIR:-$PROJECT_DIR/logs/autonomous_session_watch}"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

# One pointer line per day into the crontab redirect file: everything below
# runs inside a block redirected to $LOG_FILE, so without this the redirect
# file stays empty forever and reads as "the watcher never ran" (task #580
# item-3 diagnosis, 2026-06-12).
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

# Shared-VM HF cache redirect (#1369): the hf-hub-ttl eviction pass reads
# HF_HUB_CACHE from env (autonomous_session_watch.py has no load_dotenv) —
# point it at the LIVE data-disk cache so eviction manages the cache new
# work actually writes to. Guarded on the dir so non-VM checkouts no-op.
_eps_hf_cache_root="/mnt/eps-data/$(id -un)/huggingface-cache"
if [ -d "$_eps_hf_cache_root" ]; then
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$_eps_hf_cache_root/hub}"
  export HF_XET_CACHE="${HF_XET_CACHE:-$_eps_hf_cache_root/xet}"
fi

{
    echo "=== $(date -Iseconds) autonomous_session_watch start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/autonomous_session_watch.py
    rc=$?
    echo "=== $(date -Iseconds) autonomous_session_watch exit=$rc ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) autonomous_session_watch: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 regardless — the log file is the audit trail; we don't want cron email
# on every routine "all sessions alive" pass or transient respawn.
exit 0

# SESSION-RECONCILE AUTO-STOP IS THE DEFAULT (user request, 2026-06-10: "Can
# we stop the happy sessions once they reach awaiting promotion?" — this
# supersedes the same-day alert-only decision; 73 idle registered sessions
# had accumulated ~35-40GB RSS). No env export is needed here. To fall back
# to alert-only, export EPM_SESSION_RECONCILE_AUTOSTOP=0 in the crontab line.
