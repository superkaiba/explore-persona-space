#!/bin/bash
# Heartbeat: alert if last night's /daily file never landed (task #711), is
# stale, or is a never-enriched stub-first husk (task #1189).
# A silently-failed nightly /daily reads as "a quiet day" — the consolidation
# does not happen and nothing alerts. This cron checks logs/daily/<yesterday>.md
# and Telegram-pushes if it is MISSING, older than 25h, or present+fresh but a
# HUSK (the '## Applied workflow improvements' section missing or empty —
# under stub-first the skeleton exists from run start, so existence/mtime
# alone can no longer prove the nightly enrichment ran), so a failed /daily is
# visible instead of invisible.
#
# Output lives at logs/daily_healthcheck/YYYY-MM-DD.log (one file per day).
# Alert-once: a date-stamped sentinel logs/daily_healthcheck/sent-<yesterday>.flag
# suppresses re-alerting the same missing day (belt-and-suspenders against a
# manual re-run or a future hourly cadence).

set -uo pipefail

# Keep uv on PATH for shape-consistency with the other cron wrappers (this
# wrapper does no python work, but the guard is harmless and catches a broken
# environment early).
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); environment broken" >&2
    exit 1
fi

PROJECT_DIR="${EPS_HEALTHCHECK_PROJECT_DIR:-/home/thomasjiralerspong/explore-persona-space}"
DATE=$(date +%Y-%m-%d)
# Overridable for tests; defaults to GNU `date -d yesterday`.
YESTERDAY="${EPS_HEALTHCHECK_YESTERDAY:-$(date -d 'yesterday' +%Y-%m-%d)}"
DAILY_DIR="${EPS_HEALTHCHECK_DAILY_DIR:-$PROJECT_DIR/logs/daily}"
SENTINEL_DIR="${EPS_HEALTHCHECK_SENTINEL_DIR:-$PROJECT_DIR/logs/daily_healthcheck}"
LOG_DIR="${EPS_HEALTHCHECK_LOG_DIR:-$PROJECT_DIR/logs/daily_healthcheck}"
TELEGRAM_PUSH="${EPS_TELEGRAM_PUSH_SCRIPT:-$HOME/my-goat/scripts/telegram_push.sh}"
LOG_FILE="$LOG_DIR/$DATE.log"
DAILY_FILE="$DAILY_DIR/$YESTERDAY.md"
SENTINEL="$SENTINEL_DIR/sent-$YESTERDAY.flag"

mkdir -p "$LOG_DIR" "$SENTINEL_DIR"

FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

{
    echo "=== $(date -Iseconds) daily_healthcheck start (yesterday=$YESTERDAY) ==="

    # Missing OR stale (mtime older than 25h = 1500 min; 25h absorbs cron jitter
    # between the 23:27 /daily start and the 06:00 healthcheck — moved from
    # 01:00 in #994: the 3h bg-wait ceiling makes post-01:00 completion legitimate)
    # OR a husk (#1189: present + fresh but never enriched).
    NEEDS_ALERT=0
    ALERT_CLASS=""
    if [ ! -f "$DAILY_FILE" ]; then
        echo "daily_healthcheck: $DAILY_FILE MISSING"
        NEEDS_ALERT=1; ALERT_CLASS="missing"
    elif [ -n "$(find "$DAILY_FILE" -mmin +1500 2>/dev/null)" ]; then
        echo "daily_healthcheck: $DAILY_FILE present but mtime > 25h (stale)"
        NEEDS_ALERT=1; ALERT_CLASS="stale"
    elif ! awk '/^## Applied workflow improvements[[:space:]]*$/{flag=1; next} /^## /{flag=0} flag && NF {found=1; exit} END{exit !found}' "$DAILY_FILE"; then
        # Present + fresh but a HUSK: the stub-first skeleton (#1189) exists
        # from run start, so existence/mtime alone can no longer prove the
        # nightly enrichment ran. The skill's own done-predicate is a
        # non-empty '## Applied workflow improvements' section (SKILL.md
        # § Output refuse rule); missing H2 or empty section => never enriched.
        # Timing: nightly starts 23:27 with a 3h bg-wait ceiling (~02:30 done);
        # this check runs ~06:00, so a still-empty Applied section has >=3.5h
        # margin and IS the failure signature. If the cron start, the bg-wait
        # ceiling, or the healthcheck hour ever change such that a healthy run
        # can still be enriching at check time, revisit this arm.
        echo "daily_healthcheck: $DAILY_FILE present+fresh but HUSK (missing/empty '## Applied workflow improvements' — stub never enriched)"
        NEEDS_ALERT=1; ALERT_CLASS="husk"
    else
        echo "daily_healthcheck: $DAILY_FILE present, fresh, and enriched — OK"
    fi

    if [ "$NEEDS_ALERT" = 1 ]; then
        if [ -f "$SENTINEL" ]; then
            echo "daily_healthcheck: sentinel $SENTINEL already exists — skipping re-alert"
        else
            # $HOME is expanded by THIS shell at push-time into the real absolute
            # path before the message string is handed to telegram_push.sh (which
            # does NO expansion of the message body). A literal ~ would arrive
            # verbatim on the phone. One shared per-date sentinel regardless of
            # ALERT_CLASS — one broken day buzzes once (a missing->husk
            # transition after an alert adds no information worth a second buzz).
            if [ "$ALERT_CLASS" = "husk" ]; then
                MSG="ALERT: /daily for $YESTERDAY left only a stub (empty '## Applied workflow improvements' — run died before enrichment) — check $HOME/my-goat/logs/daily_retrospective.log | backfill: cd $PROJECT_DIR && CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=10800000 $HOME/.local/bin/claude -p '/daily $YESTERDAY' (Edit-in-place recovery: SKILL.md § Output empty-Applied branch)"
            else
                MSG="ALERT: /daily for $YESTERDAY did not land — check $HOME/my-goat/logs/daily_retrospective.log | backfill: cd $PROJECT_DIR && CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=10800000 $HOME/.local/bin/claude -p '/daily $YESTERDAY' (see .claude/skills/daily/SKILL.md § Backfill a missed day)"
            fi
            if [ -x "$TELEGRAM_PUSH" ]; then
                if "$TELEGRAM_PUSH" "$MSG"; then
                    touch "$SENTINEL"
                    echo "daily_healthcheck: alert pushed + sentinel written ($SENTINEL)"
                else
                    echo "daily_healthcheck: telegram_push.sh FAILED (no sentinel written; will retry next run)"
                fi
            else
                echo "daily_healthcheck: telegram_push.sh not executable at $TELEGRAM_PUSH — cannot alert"
            fi
        fi
    fi

    echo "=== $(date -Iseconds) daily_healthcheck end ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) daily_healthcheck: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 regardless — a best-effort heartbeat; a telegram send-fail is logged
# (and re-tried next run) but never produces a cron email.
exit 0
