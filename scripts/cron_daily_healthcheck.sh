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
# Auto-backfill (task #2113): on a MISSING or HUSK detection the healthcheck
# ADDITIONALLY launches the backfill it names — detached (setsid + nohup, own
# log at logs/daily_healthcheck/backfill-<date>.log), single-flight (flock -n
# on backfill.lock), exactly ONE auto-attempt per date
# (backfill-attempt-<date>.flag; a same-day re-run never relaunches). `stale`
# stays alert-only (a stale file may already be enriched; auto-backfilling it
# is ambiguous). A LATER run that finds an attempted date still missing/husk
# pushes a one-time "auto-backfill FAILED" alert
# (backfill-failed-sent-<date>.flag) and never relaunches — recovery from a
# failed auto-attempt is the manual command the FAILED alert names.
# Env knobs:
#   EPS_HEALTHCHECK_AUTO_BACKFILL (default 1) — 0 disables the LAUNCH only;
#     alert messages revert to the manual-command form. The failure-
#     verification sweep stays live either way (it only acts on attempt flags
#     a prior ENABLED run wrote, so with =0 from day one behavior matches
#     today's byte-for-byte).
#   EPS_HEALTHCHECK_CLAUDE_BIN (default $HOME/.local/bin/claude) — the binary
#     the backfill launches; tests point this at a fake recording script.
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
AUTO_BACKFILL="${EPS_HEALTHCHECK_AUTO_BACKFILL:-1}"
CLAUDE_BIN="${EPS_HEALTHCHECK_CLAUDE_BIN:-$HOME/.local/bin/claude}"
LOG_FILE="$LOG_DIR/$DATE.log"
DAILY_FILE="$DAILY_DIR/$YESTERDAY.md"
SENTINEL="$SENTINEL_DIR/sent-$YESTERDAY.flag"
BF_ATTEMPT="$SENTINEL_DIR/backfill-attempt-$YESTERDAY.flag"
BF_LOG="$SENTINEL_DIR/backfill-$YESTERDAY.log"
BF_LOCK="$SENTINEL_DIR/backfill.lock"

mkdir -p "$LOG_DIR" "$SENTINEL_DIR"

# Shared missing-or-husk predicate (#2113): returns 0 when the daily file at
# $1 is MISSING or a HUSK (its '## Applied workflow improvements' section
# missing or empty — the /daily done-predicate, SKILL.md § Output). Used by
# the yesterday husk arm AND the failure-verification sweep. No staleness
# read — the sweep deliberately keys on content only.
is_missing_or_husk() {
    [ ! -f "$1" ] && return 0
    ! awk '/^## Applied workflow improvements[[:space:]]*$/{flag=1; next} /^## /{flag=0} flag && NF {found=1; exit} END{exit !found}' "$1"
}

FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

{
    echo "=== $(date -Iseconds) daily_healthcheck start (yesterday=$YESTERDAY) ==="

    # ── Failure-verification sweep (#2113) ────────────────────────────────
    # Re-check every date that ever got an auto-backfill attempt (glob-driven
    # — a multi-day cron outage cannot slide a failed date out of scope).
    # SKIPS $YESTERDAY's own flag: a same-morning attempt may still be
    # running; it is re-checked from the next day onward. One FAILED alert
    # per date (its own backfill-failed-sent flag — the shared sent-<date>
    # alert sentinel does NOT suppress this distinct alert); NEVER relaunches.
    # Runs BEFORE the yesterday check so its pushes are independent of
    # NEEDS_ALERT, and regardless of EPS_HEALTHCHECK_AUTO_BACKFILL (it only
    # acts on flags a prior ENABLED run wrote).
    for BF_FLAG in "$SENTINEL_DIR"/backfill-attempt-*.flag; do
        [ -e "$BF_FLAG" ] || continue   # nullglob-safe: skip the literal glob on no match
        BF_DATE="${BF_FLAG##*/backfill-attempt-}"
        BF_DATE="${BF_DATE%.flag}"
        if [ "$BF_DATE" = "$YESTERDAY" ]; then
            continue    # in-flight same-morning attempt — not judged failed today
        fi
        FAILED_SENT="$SENTINEL_DIR/backfill-failed-sent-$BF_DATE.flag"
        if [ -f "$FAILED_SENT" ]; then
            continue    # already alerted once for this date
        fi
        if is_missing_or_husk "$DAILY_DIR/$BF_DATE.md"; then
            FAIL_MSG="ALERT: auto-backfill for $BF_DATE FAILED (day still unmined) — check $SENTINEL_DIR/backfill-$BF_DATE.log | manual backfill: cd $PROJECT_DIR && CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=10800000 $CLAUDE_BIN -p '/daily $BF_DATE'"
            if [ -x "$TELEGRAM_PUSH" ]; then
                if "$TELEGRAM_PUSH" "$FAIL_MSG"; then
                    touch "$FAILED_SENT"
                    echo "daily_healthcheck: failed-backfill alert pushed for $BF_DATE + sentinel written ($FAILED_SENT)"
                else
                    echo "daily_healthcheck: telegram_push.sh FAILED for failed-backfill alert ($BF_DATE) — will retry next run"
                fi
            else
                echo "daily_healthcheck: telegram_push.sh not executable at $TELEGRAM_PUSH — cannot alert failed backfill for $BF_DATE"
            fi
        else
            echo "daily_healthcheck: auto-backfill for $BF_DATE recovered (file enriched) — no failure alert"
        fi
    done

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
    elif is_missing_or_husk "$DAILY_FILE"; then
        # Present + fresh but a HUSK: the stub-first skeleton (#1189) exists
        # from run start, so existence/mtime alone can no longer prove the
        # nightly enrichment ran. The skill's own done-predicate is a
        # non-empty '## Applied workflow improvements' section (SKILL.md
        # § Output refuse rule); missing H2 or empty section => never enriched
        # (the file exists at this elif, so is_missing_or_husk == husk here).
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
        # ── Auto-backfill launch (#2113): missing/husk only ───────────────
        # Runs BEFORE the alert-sentinel check: the one-attempt-per-date
        # semantics belong to the attempt flag, not the alert sentinel (a
        # sent-flag written by a pre-#2113 run must not block the recovery).
        BF_STATUS=""
        if [ "$AUTO_BACKFILL" = 1 ] && { [ "$ALERT_CLASS" = "missing" ] || [ "$ALERT_CLASS" = "husk" ]; }; then
            if [ -f "$BF_ATTEMPT" ]; then
                BF_STATUS="already-attempted"
                echo "daily_healthcheck: auto-backfill already attempted for $YESTERDAY ($BF_ATTEMPT) — not relaunching"
            elif [ ! -x "$CLAUDE_BIN" ]; then
                echo "daily_healthcheck: claude bin not executable at $CLAUDE_BIN — cannot auto-backfill"
            else
                # Detached + single-flight: setsid survives this cron's exit;
                # flock -n guarantees at most one auto-backfill process
                # fleet-wide (a held lock exits 1 into BF_LOG, caught later by
                # the failure sweep); timeout bounds a wedged run (4h > the 3h
                # bg-wait ceiling the command itself sets). The attempt flag
                # is written even if the detached flock loses the race (plan
                # D4): the failure sweep catches a lock-blocked attempt on a
                # later run, keeping the parent-side logic sentinel-simple.
                BF_PID=$(bash -c "cd '$PROJECT_DIR' && setsid nohup flock -n '$BF_LOCK' \
                    timeout --kill-after=300s 14400s \
                    env CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=10800000 \
                    '$CLAUDE_BIN' -p '/daily $YESTERDAY' < /dev/null >> '$BF_LOG' 2>&1 & echo \$!")
                touch "$BF_ATTEMPT"
                BF_STATUS="launched"
                echo "daily_healthcheck: auto-backfill launched for $YESTERDAY (pid=$BF_PID, log=$BF_LOG)"
            fi
        fi

        if [ -f "$SENTINEL" ]; then
            echo "daily_healthcheck: sentinel $SENTINEL already exists — skipping re-alert"
        else
            # $HOME is expanded by THIS shell at push-time into the real absolute
            # path before the message string is handed to telegram_push.sh (which
            # does NO expansion of the message body). A literal ~ would arrive
            # verbatim on the phone. One shared per-date sentinel regardless of
            # ALERT_CLASS — one broken day buzzes once (a missing->husk
            # transition after an alert adds no information worth a second buzz).
            # Backfill segment (#2113): the launched / already-attempted forms
            # when the auto-launch path engaged; otherwise the manual paste-
            # ready command (stale class, kill switch off, claude bin missing).
            BF_CMD="cd $PROJECT_DIR && CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=10800000 $CLAUDE_BIN -p '/daily $YESTERDAY'"
            if [ "$BF_STATUS" = "launched" ]; then
                BF_PART="auto-backfill launched (attempt 1); log: $BF_LOG"
            elif [ "$BF_STATUS" = "already-attempted" ]; then
                BF_PART="auto-backfill already attempted (log: $BF_LOG); manual backfill: $BF_CMD"
            else
                BF_PART="backfill: $BF_CMD"
            fi
            if [ "$ALERT_CLASS" = "husk" ]; then
                MSG="ALERT: /daily for $YESTERDAY left only a stub (empty '## Applied workflow improvements' — run died before enrichment) — check $HOME/my-goat/logs/daily_retrospective.log | $BF_PART (Edit-in-place recovery: SKILL.md § Output empty-Applied branch)"
            else
                MSG="ALERT: /daily for $YESTERDAY did not land — check $HOME/my-goat/logs/daily_retrospective.log | $BF_PART (see .claude/skills/daily/SKILL.md § Backfill a missed day)"
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
