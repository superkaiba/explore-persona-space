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
#
# Exit-3 budget-refusal alert (task #2190): when the consolidator returns rc=3
# (#2189 — gotcha promotion refused because appending would push
# .claude/rules/gotchas.md past GOTCHAS_SIZE_WARN_BYTES), this wrapper
# Telegram-pushes ONE alert per calendar day (per-date sentinel
# refused-<date>.flag under the sentinel dir; a FAILED push writes no sentinel
# so the next pass retries) and appends one JSON row to the audit sidecar.
# The sidecar is AUDIT-ONLY — no watcher pass reads it; the push is the live
# notification channel (cron email is structurally dead on this VM: no MTA,
# and the crontab line redirects 2>&1). rc=0 passes stay silent and the
# unconditional `exit 0` below is retained.
# Env knobs:
#   EPS_LESSON_CONSOLIDATE_LOG_DIR (default $PROJECT_DIR/logs/lesson_consolidate)
#   EPS_LESSON_CONSOLIDATE_SENTINEL_DIR (default: the log dir)
#   EPS_LESSON_CONSOLIDATE_SIDECAR
#     (default $PROJECT_DIR/.claude/cache/lesson-consolidate-events.jsonl)
#   EPS_TELEGRAM_PUSH_SCRIPT (default $HOME/my-goat/scripts/telegram_push.sh)
#   EPS_LESSON_CONSOLIDATE_BIN — TEST-ONLY consolidator override (default empty
#     = the real `uv run python scripts/consolidate_lessons.py`); never set in
#     the real environment or the crontab line.

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
SENTINEL_DIR="${EPS_LESSON_CONSOLIDATE_SENTINEL_DIR:-$LOG_DIR}"
TELEGRAM_PUSH="${EPS_TELEGRAM_PUSH_SCRIPT:-$HOME/my-goat/scripts/telegram_push.sh}"
SIDECAR="${EPS_LESSON_CONSOLIDATE_SIDECAR:-$PROJECT_DIR/.claude/cache/lesson-consolidate-events.jsonl}"
SENTINEL="$SENTINEL_DIR/refused-$DATE.flag"

mkdir -p "$LOG_DIR" "$SENTINEL_DIR"

# One pointer line per day into the crontab redirect file: everything below
# runs inside a block redirected to $LOG_FILE, so without this the redirect
# file stays empty forever and reads as "the cron never ran" (task #580
# item-3 diagnosis; mirrors cron_pod_audit.sh).
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

{
    echo "=== $(date -Iseconds) lesson_consolidate start ==="
    cd "$PROJECT_DIR" || exit 1
    # Test seam (mirrors EPS_HEALTHCHECK_CLAUDE_BIN in cron_daily_healthcheck.sh):
    # default empty, so the production path is unchanged.
    if [ -n "${EPS_LESSON_CONSOLIDATE_BIN:-}" ]; then
        "$EPS_LESSON_CONSOLIDATE_BIN" --apply --window-days 7
    else
        uv run python scripts/consolidate_lessons.py --apply --window-days 7
    fi
    rc=$?
    echo "=== $(date -Iseconds) lesson_consolidate exit=$rc ==="
} >> "$LOG_FILE" 2>&1

# rc=3 = promote_refused_budget (#2189): gotcha promotion refused because the
# append would push gotchas.md past its byte budget. Surface it loud — keyed
# STRICTLY on rc 3 (any other non-zero rc is a generic crash, not a budget
# refusal; do NOT broaden to -ne 0). `rc` is live here because the block above
# is a brace group, not a subshell.
#
# ${rc:-0} — the ONE path that leaves rc unset is an unwritable $LOG_FILE (an
# uncreatable LOG_DIR, ENOSPC): the brace group's redirect fails, so the group
# never runs and never assigns rc. A bare "$rc" would then trip `set -u`
# ("rc: unbound variable", exit 1) where the pre-diff wrapper exited 0 —
# an unintended behaviour change from a change that is only supposed to ADD an
# alert path. Defaulting to 0 keeps that path byte-identical to pre-diff
# behaviour. (The wrapper being silent when its own log dir is uncreatable is a
# real, PRE-EXISTING gap — filed separately, not widened here.)
if [ "${rc:-0}" -eq 3 ]; then
    {
        # Parse the refused-bullet count from the consolidator's stderr INFO
        # summary line, captured into $LOG_FILE by the brace group's 2>&1 —
        # NOT from the consolidator's own counts-line file, which is UTC-dated
        # and lands in the NEXT day's file at the 23:50-local cron hour
        # (#2189 dating mismatch). `tail -1` takes the LAST match = the
        # current run's line; a missing or format-drifted line degrades to the
        # literal `unknown` — never a crash, never an empty field.
        REFUSED=$(grep -o 'promote_refused_budget=[0-9]*' "$LOG_FILE" 2>/dev/null \
                  | tail -1 | cut -d= -f2)
        [ -n "$REFUSED" ] || REFUSED=unknown
        MSG="ALERT: lesson_consolidate refused $REFUSED gotcha bullet(s) — .claude/rules/gotchas.md is at its byte budget (GOTCHAS_SIZE_WARN_BYTES). No write, no commit; the same bullets re-refuse nightly until gotchas.md is re-trimmed. Log: $LOG_FILE (refused bullets printed verbatim there)"
        # Audit sidecar row BEFORE the sentinel check: a suppressed re-alert
        # still leaves a row (the sentinel dedups the buzz, not the record).
        # Sidecar failure is non-fatal — the push is the live channel (§2.4).
        printf '{"ts":"%s","event":"promote_refused_budget","refused":"%s","log":"%s","rc":3}\n' \
            "$(date -Iseconds)" "$REFUSED" "$LOG_FILE" >> "$SIDECAR" 2>/dev/null \
            || echo "lesson_consolidate: sidecar append failed ($SIDECAR) — push is the live channel, continuing"
        if [ -f "$SENTINEL" ]; then
            echo "lesson_consolidate: sentinel $SENTINEL already exists — skipping re-alert"
        elif [ -x "$TELEGRAM_PUSH" ]; then
            if "$TELEGRAM_PUSH" "$MSG"; then
                touch "$SENTINEL"
                echo "lesson_consolidate: budget-refusal alert pushed + sentinel written ($SENTINEL)"
            else
                echo "lesson_consolidate: telegram_push.sh FAILED (no sentinel written; will retry next run)"
            fi
        else
            echo "lesson_consolidate: telegram_push.sh not executable at $TELEGRAM_PUSH — cannot alert"
        fi
    } >> "$LOG_FILE" 2>&1
fi

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) lesson_consolidate: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 regardless — the log file is the audit trail, no cron email per routine
# pass (and none would be delivered anyway: no MTA + the crontab 2>&1 redirect).
# The rc=3 arm above is the loud channel.
exit 0
