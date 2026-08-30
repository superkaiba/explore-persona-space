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
# We DO pass `--force` to `prune` (lock override only — see the invocation
# below for the measured reason). `prune --force` != `clean --force`.
#
# vm_disk_guard.py ALSO runs `uv cache prune` as its tier-(a) cleanup, but only
# when / crosses the 85% threshold; this standalone idle-time cron keeps the
# cache trimmed continuously so the guard rarely has to fire on a full disk.
#
# Recommended schedule (NOT registered here — the orchestrator registers crons):
#   daily off-peak, e.g.  17 4 * * *   (04:17 PT, alongside the other dailies)
#
# Output lives at logs/uv_cache_prune/YYYY-MM-DD.log (one file per day).
# Fails loud (stderr FATAL + exit 1) when the log dir cannot be created or the
# daily log file is not appendable (task #2386; ports the #2196 pattern).

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
    || fatal "cannot create log dir (LOG_DIR=$LOG_DIR); uv cache prune NOT run"

# One pointer line per day into the crontab redirect file (mirrors
# cron_vm_disk_guard.sh): everything below runs inside a block redirected to
# $LOG_FILE, so without this the redirect file reads as "the cron never ran".
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

# mkdir -p succeeds on an existing dir regardless of writability, so probe the
# actual append open the brace group below will attempt (#2196 ordering: after
# the FIRST_RUN_OF_DAY read — the probe creates $LOG_FILE when absent).
: >> "$LOG_FILE" 2>/dev/null \
    || fatal "daily log file not appendable ($LOG_FILE); uv cache prune NOT run"

{
    echo "=== $(date -Iseconds) uv_cache_prune start ==="
    cd "$PROJECT_DIR" || exit 1
    # `--force` overrides the cache LOCK. This is NOT the `uv cache clean
    # --force` the header warns against: `clean` wipes the whole cache,
    # `prune --force` still removes only entries no longer referenced by any
    # installed environment — it just declines to wait for the lock.
    #
    # Why it is required (measured 2026-08-16): the lock is held continuously
    # by long-lived `uv tool uvx` MCP servers (arxiv-mcp-server /
    # arxiv-latex-mcp) belonging to LIVE Claude sessions — 157 of 165 such
    # processes were session-owned. Their venvs live in
    # ~/.local/share/uv/tools/, NOT the cache, so pruning cannot strand them.
    # A box that always has a Claude session up can therefore NEVER win the
    # lock: without --force this cron had failed rc=2 on the 300s timeout
    # every night (2026-08-14/15/16 logs all identical), and the cache grew
    # unpruned until a manual run reclaimed 6.6 GiB.
    #
    # UV_LOCK_TIMEOUT is dropped from 300s to 30s: with --force the wait is
    # pointless, and a short timeout keeps the cron off the disk-guard's back.
    UV_LOCK_TIMEOUT=30 uv cache prune --force
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "uv cache prune --force returned rc=$rc (unexpected — --force overrides the lock); will retry next pass"
    fi
    echo "=== $(date -Iseconds) uv_cache_prune exit=$rc ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) uv_cache_prune: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Always exit 0 — a held-lock prune failure is transient (the next pass
# retries) and must not spam a cron email. The log file is the audit trail.
exit 0
