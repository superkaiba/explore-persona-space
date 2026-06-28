#!/bin/bash
# Scheduled `uv cache prune` — invoked from system crontab (idle-time).
#
# The global uv package cache grows unbounded as dependency sets churn across
# sessions and pods; nothing prunes it on a schedule, so it ballooned to ~56 GB
# on the fleet-shared VM root disk by 2026-06-26 (a standing contributor to the
# / disk-full incidents). `uv cache prune` removes cache entries no longer
# referenced by any installed environment — it KEEPS in-use entries, so it is
# safe to run unconditionally and never strands a live env. We deliberately do
# NOT use `uv cache clean --force` (that wipes the WHOLE cache, forcing a full
# re-download on the next `uv run` everywhere).
#
# vm_disk_guard.py ALSO runs `uv cache prune` as its tier-(a) cleanup, but only
# when / crosses the 85% threshold; this standalone idle-time cron keeps the
# cache trimmed continuously so the guard rarely has to fire on a full disk.
#
# Recommended schedule (NOT registered here — the orchestrator registers crons):
#   daily off-peak, e.g.  17 4 * * *   (04:17 PT, alongside the other dailies)
#
# Output lives at logs/uv_cache_prune/YYYY-MM-DD.log (one file per day).

set -uo pipefail

# cron's minimal PATH lacks ~/.local/bin, so a bare `uv` exit-127s silently.
# Put uv on PATH; fail LOUD if still missing (a silent no-op would read as
# "the cache is fine" while it keeps growing).
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); cannot prune uv cache" >&2
    exit 1
fi

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="${EPS_UV_CACHE_PRUNE_LOG_DIR:-$PROJECT_DIR/logs/uv_cache_prune}"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

# One pointer line per day into the crontab redirect file (mirrors
# cron_vm_disk_guard.sh): everything below runs inside a block redirected to
# $LOG_FILE, so without this the redirect file reads as "the cron never ran".
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

{
    echo "=== $(date -Iseconds) uv_cache_prune start ==="
    cd "$PROJECT_DIR" || exit 1
    # `uv cache prune` is graceful when the cache lock is held by a concurrent
    # `uv run` (it returns non-zero rather than corrupting state); we log the
    # rc and never let a transient lock failure raise a cron email.
    uv cache prune
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "uv cache prune returned rc=$rc (likely a held cache lock); will retry next pass"
    fi
    echo "=== $(date -Iseconds) uv_cache_prune exit=$rc ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) uv_cache_prune: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Always exit 0 — a held-lock prune failure is transient (the next pass
# retries) and must not spam a cron email. The log file is the audit trail.
exit 0
