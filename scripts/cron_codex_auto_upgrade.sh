#!/bin/bash
# Daily Codex CLI + twin-model auto-upgrade — invoked from the system crontab.
#
# Keeps `codex` at the newest npm release and the twin's model at the newest
# slug that CLI can actually run, then restarts the app-server so the running
# runtime is not the pre-upgrade one. Full rationale + the three failure modes
# it automates away: scripts/codex_auto_upgrade.py module docstring.
#
# Recommended schedule: 17 7 * * *  (off-minute PT, before the workday, and
# clear of the 05:31 step9c ledger refresh + the 09:37/09:47 audit crons).
#
# The upgrader owns its own safety semantics and they are NOT reimplemented
# here: it aborts when any Codex job is in flight (re-checked immediately
# before the app-server kill), probes a candidate model with a real call
# before writing it to config, and never touches model_reasoning_effort.
#
# Output lives at logs/codex_auto_upgrade/YYYY-MM-DD.log (one file per day).
#
# Non-zero-rc alert (mirrors cron_step9c_ledger_refresh.sh): on rc != 0 this
# wrapper Telegram-pushes ONE alert per calendar day (per-date sentinel
# failed-<date>.flag under the sentinel dir; a FAILED push writes no sentinel
# so the next pass retries) and appends one JSON row to the audit sidecar.
# The push is the live channel — cron email is structurally dead on this VM
# (no MTA, and the crontab line redirects 2>&1). rc=0 passes stay silent and
# the unconditional `exit 0` below is retained. The prerequisite preflight
# and the log/sentinel-dir mkdir route through the SAME rc != 0 arm (they
# set rc=1 and skip the upgrader) — a bare `exit` before the alert variables
# existed was a structurally silent failure, the same class as the failed-cd
# fix inside the brace group below.
#
# Env knobs:
#   EPS_CODEX_UPGRADE_LOG_DIR (default $PROJECT_DIR/logs/codex_auto_upgrade)
#   EPS_CODEX_UPGRADE_SENTINEL_DIR (default: the log dir)
#   EPS_CODEX_UPGRADE_SIDECAR
#     (default $PROJECT_DIR/.claude/cache/codex-auto-upgrade-events.jsonl)
#   EPS_TELEGRAM_PUSH_SCRIPT (default $HOME/my-goat/scripts/telegram_push.sh)
#   EPS_CODEX_UPGRADE_BIN — TEST-ONLY upgrader override (default empty = the
#     real `uv run python scripts/codex_auto_upgrade.py`); never set in the
#     real environment or the crontab line.

set -uo pipefail

# cron's minimal PATH lacks ~/.local/bin (uv) and ~/.npm-global/bin (codex,
# and the npm-global prefix `npm install -g` writes to), so bare invocations
# exit-127 silently behind the `exit 0` below.
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:/usr/local/bin:$PATH"

# PROJECT_DIR is deliberately hardcoded — no env override, so the crontab
# environment can never point this wrapper at an arbitrary checkout.
PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="${EPS_CODEX_UPGRADE_LOG_DIR:-$PROJECT_DIR/logs/codex_auto_upgrade}"
LOG_FILE="$LOG_DIR/$DATE.log"
SENTINEL_DIR="${EPS_CODEX_UPGRADE_SENTINEL_DIR:-$LOG_DIR}"
TELEGRAM_PUSH="${EPS_TELEGRAM_PUSH_SCRIPT:-$HOME/my-goat/scripts/telegram_push.sh}"
SIDECAR="${EPS_CODEX_UPGRADE_SIDECAR:-$PROJECT_DIR/.claude/cache/codex-auto-upgrade-events.jsonl}"
SENTINEL="$SENTINEL_DIR/failed-$DATE.flag"

# log_line <text>: append to $LOG_FILE when it is writable, else stderr. The
# alert path must never lose its own diagnostics to the exact failure it is
# reporting — a redirect onto an unwritable $LOG_FILE silently skips the
# redirected command.
log_line() {
    if ! echo "$1" >> "$LOG_FILE" 2>/dev/null; then
        echo "$1" >&2
    fi
}

# alert_failure <rc>: the ONE alert path for every wrapper failure — audit
# sidecar row first (the sentinel dedups the buzz, not the record), then the
# per-date-deduped Telegram push; a FAILED push writes no sentinel so the
# next pass retries. Called only from the rc != 0 arm below.
alert_failure() {
    local rc="$1"
    local msg="ALERT: codex_auto_upgrade FAILED (rc=$rc) — the Codex CLI/twin-model auto-upgrade did not complete. The twin may be on a stale runtime or an unusable model; check with a codex_task.py probe. Log: $LOG_FILE"
    printf '{"ts":"%s","event":"upgrade_failed","rc":%s,"log":"%s"}\n' \
        "$(date -Iseconds)" "$rc" "$LOG_FILE" >> "$SIDECAR" 2>/dev/null \
        || log_line "codex_auto_upgrade: sidecar append failed ($SIDECAR) — push is the live channel, continuing"
    if [ -f "$SENTINEL" ]; then
        log_line "codex_auto_upgrade: sentinel $SENTINEL already exists — skipping re-alert"
    elif [ -x "$TELEGRAM_PUSH" ]; then
        local push_out push_rc
        push_out=$("$TELEGRAM_PUSH" "$msg" 2>&1)
        push_rc=$?
        [ -n "$push_out" ] && log_line "$push_out"
        if [ "$push_rc" -eq 0 ]; then
            if touch "$SENTINEL" 2>/dev/null; then
                log_line "codex_auto_upgrade: failure alert pushed + sentinel written ($SENTINEL)"
            else
                log_line "codex_auto_upgrade: failure alert pushed; sentinel write FAILED ($SENTINEL) — next pass may re-alert"
            fi
        else
            log_line "codex_auto_upgrade: telegram_push.sh FAILED (no sentinel written; will retry next run)"
        fi
    else
        log_line "codex_auto_upgrade: telegram_push.sh not executable at $TELEGRAM_PUSH — cannot alert"
    fi
}

SETUP_OK=1

# An uncreatable log/sentinel dir must fail LOUD: unchecked, the brace
# group's `>> "$LOG_FILE"` redirect fails, the group never runs, rc is never
# assigned, and ${rc:-0} converts the failure into silent success. Checked
# BEFORE the prerequisite preflight so preflight diagnostics can land in the
# log file when the dirs are creatable.
if ! mkdir_err=$(mkdir -p "$LOG_DIR" "$SENTINEL_DIR" 2>&1); then
    log_line "$(date -Iseconds) FATAL: cannot create log/sentinel dirs ($mkdir_err) — skipping upgrader"
    SETUP_OK=0
fi

# A missing prerequisite must NOT `exit` here: a bare exit skips the rc != 0
# alert arm below, and with no MTA + the crontab redirecting stderr that is a
# completely silent failure. Record the failure, skip the upgrader, and let
# rc=1 fall through to the alert arm.
for bin in uv npm codex; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        log_line "$(date -Iseconds) FATAL: $bin not on PATH ($PATH); cannot auto-upgrade codex"
        SETUP_OK=0
    fi
done

# One pointer line per day into the crontab redirect file: everything below
# runs inside a block redirected to $LOG_FILE, so without this the redirect
# file stays empty forever and reads as "the cron never ran".
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

# mkdir -p returns success for an EXISTING dir regardless of writability, so
# the checked mkdir above cannot see an exists-but-unwritable log dir: the
# brace group's `>> "$LOG_FILE"` redirect then fails, the group never runs,
# rc is never assigned, and ${rc:-0} below converts the failure into a silent
# exit 0 (task #2386, Pattern C). Probe the actual append open the group will
# attempt and route the failure through SETUP_OK into the SAME alert arm as
# every other setup failure — deliberately NOT a fatal()-style early exit,
# which would skip the alert arm entirely. Ordering mirrors #2196: the probe
# CREATES $LOG_FILE when absent, so it must run AFTER the FIRST_RUN_OF_DAY
# read or the daily pointer line is lost.
if ! : >> "$LOG_FILE" 2>/dev/null; then
    log_line "$(date -Iseconds) FATAL: daily log file not appendable ($LOG_FILE) — skipping upgrader"
    SETUP_OK=0
fi

if [ "$SETUP_OK" -ne 1 ]; then
    rc=1
else
    {
        echo "=== $(date -Iseconds) codex_auto_upgrade start ==="
        # A failed cd must NOT `exit` here: this brace group is NOT a subshell,
        # so an exit would terminate the whole script BEFORE the rc != 0 alert
        # arm below — with no MTA and the crontab redirecting stderr, that is a
        # completely silent failure. Skip the upgrader instead (it is invoked by
        # a cwd-relative path, so running it from the wrong cwd would execute the
        # wrong file or fail confusingly) and let rc=1 fall through to the arm.
        if ! cd "$PROJECT_DIR"; then
            echo "$(date -Iseconds) FATAL: cd $PROJECT_DIR failed — skipping upgrader"
            rc=1
        elif [ -n "${EPS_CODEX_UPGRADE_BIN:-}" ]; then
            "$EPS_CODEX_UPGRADE_BIN"
            rc=$?
        else
            OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
                MALLOC_ARENA_MAX=2 \
                uv run python scripts/codex_auto_upgrade.py
            rc=$?
        fi
        echo "=== $(date -Iseconds) codex_auto_upgrade exit=$rc ==="
    } >> "$LOG_FILE" 2>&1
fi

# rc != 0 = a step failed: the CLI upgrade errored, the config write failed,
# or the app-server could not be restarted (leaving the twin on a stale
# runtime, which manifests as "requires a newer version of Codex" at dispatch
# and NOT as anything `codex --version` would show). Surface it loud.
#
# ${rc:-0} — a brace-group redirect that fails leaves rc unset (the group
# never runs, so it never assigns rc) and a bare "$rc" would then trip
# `set -u`. TWO of the three ways that happened are now guarded upstream and
# alert like any other setup failure: an UNCREATABLE log/sentinel dir binds
# SETUP_OK=0 through the checked mkdir, and an EXISTS-BUT-UNWRITABLE (or
# otherwise non-appendable) $LOG_FILE binds SETUP_OK=0 through the
# appendability probe (#2386). Note the checked mkdir alone does NOT cover
# the unwritable case — mkdir -p succeeds on an existing dir whatever its
# mode, which is exactly why the probe exists.
#
# The remaining residual is a genuine RACE, not a missing check: the dir is
# removed, remounted read-only, or the filesystem fills (ENOSPC) in the
# window between the probe and the brace-group redirect. ${rc:-0} is kept as
# defense-in-depth for that window (same reasoning as the #2196 wrappers).
if [ "${rc:-0}" -ne 0 ]; then
    alert_failure "$rc"
fi

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) codex_auto_upgrade: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 regardless — the log file is the audit trail, no cron email per
# routine pass. The rc != 0 arm above is the loud channel.
exit 0
