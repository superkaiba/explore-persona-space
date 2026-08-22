#!/bin/bash
# Nightly Step 9c known-red baseline-ledger refresh — invoked from system crontab (task #2114).
#
# The /issue Step 9c/10d known-red-on-main compare depends on the baseline
# ledger .claude/cache/step9c-baseline.json, which goes stale (age > 24 h or
# > 150 code-path commits) on main red waves; sessions then pay the ~31-40 min
# refresh cost mid-gate (#2105, #1992, #2106). This wrapper runs
# `scripts/step9c_baseline.py refresh --json` nightly (recommended schedule:
# 31 5 * * * — off-minute PT, while the fleet is quiet) so sessions start each
# day against a fresh ledger; in-session lazy refresh stays the fallback.
#
# Concurrency + failure semantics are the refresh command's OWN, not
# reimplemented here: the .claude/cache/step9c-baseline.lock flock makes a
# concurrent in-session refresh a single-flight no-op (rc 0), and
# timeout/junit-parse/0-collected failures are rc=2 with NO ledger write. The
# refresh's own --timeout-s default (4350 s) is the wall fence — no extra
# timeout(1) wrapper. The ledger is GITIGNORED by design (VM-local state):
# no commit/push leg.
#
# Output lives at logs/step9c_ledger_refresh/YYYY-MM-DD.log (one file per day).
#
# Non-zero-rc alert (mirrors cron_lesson_consolidate.sh): on refresh rc != 0
# this wrapper Telegram-pushes ONE alert per calendar day (per-date sentinel
# failed-<date>.flag under the sentinel dir; a FAILED push writes no sentinel
# so the next pass retries) and appends one JSON row to the audit sidecar.
# The sidecar is AUDIT-ONLY — no watcher pass reads it; the push is the live
# notification channel (cron email is structurally dead on this VM: no MTA,
# and the crontab line redirects 2>&1). rc=0 passes stay silent and the
# unconditional `exit 0` below is retained.
# Env knobs:
#   EPS_STEP9C_REFRESH_LOG_DIR (default $PROJECT_DIR/logs/step9c_ledger_refresh)
#   EPS_STEP9C_REFRESH_SENTINEL_DIR (default: the log dir)
#   EPS_STEP9C_REFRESH_SIDECAR
#     (default $PROJECT_DIR/.claude/cache/step9c-refresh-cron-events.jsonl)
#   EPS_TELEGRAM_PUSH_SCRIPT (default $HOME/my-goat/scripts/telegram_push.sh)
#   EPS_STEP9C_REFRESH_BIN — TEST-ONLY refresh override (default empty = the
#     real `uv run python scripts/step9c_baseline.py refresh --json`); never
#     set in the real environment or the crontab line.

set -uo pipefail

# cron's minimal PATH lacks ~/.local/bin, so a bare `uv` exit-127s silently
# (the `exit 0` below hides it). Put uv on PATH; fail LOUD if still missing.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); cannot refresh step9c ledger" >&2
    exit 1
fi

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="${EPS_STEP9C_REFRESH_LOG_DIR:-$PROJECT_DIR/logs/step9c_ledger_refresh}"
LOG_FILE="$LOG_DIR/$DATE.log"
SENTINEL_DIR="${EPS_STEP9C_REFRESH_SENTINEL_DIR:-$LOG_DIR}"
TELEGRAM_PUSH="${EPS_TELEGRAM_PUSH_SCRIPT:-$HOME/my-goat/scripts/telegram_push.sh}"
SIDECAR="${EPS_STEP9C_REFRESH_SIDECAR:-$PROJECT_DIR/.claude/cache/step9c-refresh-cron-events.jsonl}"
SENTINEL="$SENTINEL_DIR/failed-$DATE.flag"

mkdir -p "$LOG_DIR" "$SENTINEL_DIR"

# One pointer line per day into the crontab redirect file: everything below
# runs inside a block redirected to $LOG_FILE, so without this the redirect
# file stays empty forever and reads as "the cron never ran" (task #580
# item-3 diagnosis; mirrors cron_lesson_consolidate.sh).
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

{
    echo "=== $(date -Iseconds) step9c_ledger_refresh start ==="
    cd "$PROJECT_DIR" || exit 1
    # Fail-open self-choom: the refresh runs the heaviest pytest universe and
    # must not be the earlyoom victim; children inherit the adjustment
    # (-600, not -1000, so a runaway run still dies first). Failure tolerated.
    sudo -n choom -n -600 -p $$ >/dev/null 2>&1 \
        || echo "step9c_ledger_refresh: self-choom failed (sudo -n choom); continuing unprotected"
    # Test seam (mirrors EPS_LESSON_CONSOLIDATE_BIN in cron_lesson_consolidate.sh):
    # default empty, so the production path is unchanged. The shared-VM
    # thread-cap prefix (code-style.md #847) bounds the pytest universe's
    # BLAS/OMP pools on this fleet-shared box.
    if [ -n "${EPS_STEP9C_REFRESH_BIN:-}" ]; then
        "$EPS_STEP9C_REFRESH_BIN" refresh --json
    else
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
            MALLOC_ARENA_MAX=2 \
            uv run python scripts/step9c_baseline.py refresh --json
    fi
    rc=$?
    echo "=== $(date -Iseconds) step9c_ledger_refresh exit=$rc ==="
} >> "$LOG_FILE" 2>&1

# rc != 0 = the refresh wrote NO ledger (rc=2: pytest rc outside {0,1} /
# timeout / junit parse failure / zero collected / git or ruff failure —
# step9c_baseline.py's pinned exit codes). Surface it loud: sessions silently
# fall back to the ~31-40 min in-session lazy refresh otherwise. `rc` is live
# here because the block above is a brace group, not a subshell.
#
# ${rc:-0} — the ONE path that leaves rc unset is an unwritable $LOG_FILE (an
# uncreatable LOG_DIR, ENOSPC): the brace group's redirect fails, so the group
# never runs and never assigns rc. A bare "$rc" would then trip `set -u`
# ("rc: unbound variable", exit 1) where the template wrapper exited 0.
# Defaulting to 0 keeps that path silent-exit-0, matching
# cron_lesson_consolidate.sh.
if [ "${rc:-0}" -ne 0 ]; then
    {
        MSG="ALERT: step9c_ledger_refresh FAILED (rc=$rc) — the nightly Step 9c known-red baseline-ledger refresh wrote NO ledger (.claude/cache/step9c-baseline.json left as-is); sessions fall back to the ~31-40 min in-session lazy refresh until a refresh succeeds. Log: $LOG_FILE"
        # Audit sidecar row BEFORE the sentinel check: a suppressed re-alert
        # still leaves a row (the sentinel dedups the buzz, not the record).
        # Sidecar failure is non-fatal — the push is the live channel.
        printf '{"ts":"%s","event":"refresh_failed","rc":%s,"log":"%s"}\n' \
            "$(date -Iseconds)" "$rc" "$LOG_FILE" >> "$SIDECAR" 2>/dev/null \
            || echo "step9c_ledger_refresh: sidecar append failed ($SIDECAR) — push is the live channel, continuing"
        if [ -f "$SENTINEL" ]; then
            echo "step9c_ledger_refresh: sentinel $SENTINEL already exists — skipping re-alert"
        elif [ -x "$TELEGRAM_PUSH" ]; then
            if "$TELEGRAM_PUSH" "$MSG"; then
                touch "$SENTINEL"
                echo "step9c_ledger_refresh: refresh-failure alert pushed + sentinel written ($SENTINEL)"
            else
                echo "step9c_ledger_refresh: telegram_push.sh FAILED (no sentinel written; will retry next run)"
            fi
        else
            echo "step9c_ledger_refresh: telegram_push.sh not executable at $TELEGRAM_PUSH — cannot alert"
        fi
    } >> "$LOG_FILE" 2>&1
fi

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) step9c_ledger_refresh: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 regardless — the log file is the audit trail, no cron email per routine
# pass (and none would be delivered anyway: no MTA + the crontab 2>&1 redirect).
# The rc != 0 arm above is the loud channel.
exit 0
