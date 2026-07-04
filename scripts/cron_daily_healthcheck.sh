#!/bin/bash
# Heartbeat: alert if last night's /daily file never landed (task #711).
# A silently-failed nightly /daily reads as "a quiet day" — the consolidation
# does not happen and nothing alerts. This cron stats logs/daily/<yesterday>.md
# and Telegram-pushes if it is MISSING or older than 25h, so a failed /daily is
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
    # 01:00 in #994: the 3h bg-wait ceiling makes post-01:00 completion legitimate).
    NEEDS_ALERT=0
    if [ ! -f "$DAILY_FILE" ]; then
        echo "daily_healthcheck: $DAILY_FILE MISSING"
        NEEDS_ALERT=1
    elif [ -n "$(find "$DAILY_FILE" -mmin +1500 2>/dev/null)" ]; then
        echo "daily_healthcheck: $DAILY_FILE present but mtime > 25h (stale)"
        NEEDS_ALERT=1
    else
        echo "daily_healthcheck: $DAILY_FILE present and fresh — OK"
    fi

    if [ "$NEEDS_ALERT" = 1 ]; then
        if [ -f "$SENTINEL" ]; then
            echo "daily_healthcheck: sentinel $SENTINEL already exists — skipping re-alert"
        else
            # $HOME is expanded by THIS shell at push-time into the real absolute
            # path before the message string is handed to telegram_push.sh (which
            # does NO expansion of the message body). A literal ~ would arrive
            # verbatim on the phone.
            MSG="ALERT: /daily for $YESTERDAY did not land — check $HOME/my-goat/logs/daily_retrospective.log | backfill: cd $PROJECT_DIR && CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=10800000 $HOME/.local/bin/claude -p '/daily $YESTERDAY' (see .claude/skills/daily/SKILL.md § Backfill a missed day)"
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
