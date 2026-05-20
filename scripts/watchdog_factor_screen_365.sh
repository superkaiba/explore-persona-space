#!/usr/bin/env bash
# Round-7 (issue #365) external watchdog for the dispatcher.
#
# Survives dispatcher death by living in a separate process group (setsid)
# and respawning the dispatcher up to MAX_RESPAWNS times. Between respawns,
# checks that forward progress is being made (delta of cells with a
# non-empty metrics.json containing the "persona_panel_scores" sentinel).
#
# Round-7 hardening (vs round-6):
#   1. Dedicated watchdog log file (positional arg 4) separate from the
#      dispatcher log. log_w() writes to that file AND to stderr; the
#      launcher's `2>&1` capture preserves stderr while command
#      substitution `$(...)` (which would mix log lines into return
#      values) only captures stdout. The watchdog log is never 0 bytes
#      if the watchdog ran at all.
#   2. PID file at $WATCHDOG_PID_FILE for external liveness checks.
#      Removed on exit. Single-instance guard refuses to start if an
#      existing PID file points to a live PID.
#   3. Heartbeat line every poll interval:
#        [watchdog hh:mm:ss] alive — dispatcher-pid=N respawn=K/M complete-cells=C gap=Gs
#   4. set -euo pipefail with an ERR trap that logs the failing line.
#   5. SIGTERM/SIGHUP/SIGINT/EXIT traps that log the signal, kill the
#      dispatcher (SIGTERM, then SIGKILL after 10s), and remove the PID
#      file. The dispatcher cycle runs in the main shell (NOT a command
#      substitution subshell) so signal traps stay reachable.
#   6. count_complete_cells uses find + a count pipeline that handles the
#      zero-match case cleanly under pipefail.
#
# Usage (from inside /workspace/explore-persona-space on the pod):
#
#   setsid bash scripts/watchdog_factor_screen_365.sh \
#       /workspace/logs/issue-365-r7-dispatcher.log \
#       eval_results/issue_365 \
#       "uv run python scripts/dispatch_factor_screen_365.py \
#           --slab-root eval_results/issue_365 \
#           --pool-dir data/issue_365/pools \
#           --sources librarian,surgeon,programmer \
#           --seeds 42 \
#           --skip-pool-stage" \
#       /workspace/logs/issue-365-r7-watchdog.log \
#       < /dev/null > /dev/null 2>&1 &
#   disown
#
# (The watchdog writes its own log directly to $WATCHDOG_LOG via `>>`,
# not via stdout/stderr — so the launcher's redirections can safely
# discard both streams. log_w also mirrors to stderr for interactive
# debugging, but that copy is suppressed by `2>&1 > /dev/null` here to
# avoid double-logging if you redirect stderr to the same file.)
#
# Tail the watchdog log to see heartbeats + lifecycle events:
#   tail -f /workspace/logs/issue-365-r7-watchdog.log
# Tail the dispatcher log to see the actual Python output:
#   tail -f /workspace/logs/issue-365-r7-dispatcher.log
#
# Exits 0 when the dispatcher exits 0. Exits 1 if MAX_RESPAWNS hit without
# forward progress, or if a signal forces shutdown. Exits 2 if another
# watchdog instance is already alive.

set -euo pipefail

LOG_FILE="${1:?usage: $0 LOG_FILE SLAB_ROOT DISPATCH_CMD WATCHDOG_LOG}"
SLAB_ROOT="${2:?usage: $0 LOG_FILE SLAB_ROOT DISPATCH_CMD WATCHDOG_LOG}"
DISPATCH_CMD="${3:?usage: $0 LOG_FILE SLAB_ROOT DISPATCH_CMD WATCHDOG_LOG}"
WATCHDOG_LOG="${4:?usage: $0 LOG_FILE SLAB_ROOT DISPATCH_CMD WATCHDOG_LOG}"

MAX_RESPAWNS="${WATCHDOG_MAX_RESPAWNS:-5}"
STALL_GAP_SECONDS="${WATCHDOG_STALL_GAP_SECONDS:-1800}"  # 30 min log mtime gap
POLL_SECONDS="${WATCHDOG_POLL_SECONDS:-300}"             # 5 min poll interval
COOL_DOWN_SECONDS="${WATCHDOG_COOL_DOWN_SECONDS:-60}"
WATCHDOG_PID_FILE="${WATCHDOG_PID_FILE:-/workspace/logs/issue-365-watchdog.pid}"

# --- logging --------------------------------------------------------------

# Ensure the watchdog log file exists and is writable before any tee.
mkdir -p "$(dirname "$WATCHDOG_LOG")" 2>/dev/null || true
: >> "$WATCHDOG_LOG"

log_w() {
  # Write every watchdog message to the dedicated watchdog log AND to
  # stderr. stderr is captured by the launcher's `2>>$WATCHDOG_LOG` (so
  # the file gets the message even if some caller swallowed stdout), but
  # crucially NOT by command substitution `$(...)` — that only captures
  # stdout. The cycle function uses a global to communicate rc/before/
  # after instead of stdout, keeping stdout free of log noise.
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '[watchdog %s] %s\n' "$ts" "$*" >> "$WATCHDOG_LOG"
  printf '[watchdog %s] %s\n' "$ts" "$*" >&2
}

# --- traps ----------------------------------------------------------------

DISPATCH_PID=""
EXIT_REASON="normal"

on_err() {
  local exit_code=$?
  local line_no="${1:-?}"
  log_w "ERR trap fired: exit_code=$exit_code at line $line_no (BASH_COMMAND=${BASH_COMMAND:-?})"
  EXIT_REASON="err"
  exit "$exit_code"
}
trap 'on_err $LINENO' ERR

cleanup_dispatcher() {
  local pid="${1:-}"
  if [[ -z "$pid" ]]; then return 0; fi
  if kill -0 "$pid" 2>/dev/null; then
    log_w "cleanup: SIGTERM dispatcher pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
    # Give it 10s to exit cleanly, then escalate.
    local i
    for i in $(seq 1 10); do
      if ! kill -0 "$pid" 2>/dev/null; then return 0; fi
      sleep 1
    done
    log_w "cleanup: SIGKILL dispatcher pid=$pid (did not exit after 10s SIGTERM)"
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

on_signal() {
  local sig="$1"
  log_w "received SIG${sig} at $(date -u +%Y-%m-%dT%H:%M:%SZ); shutting down"
  EXIT_REASON="signal:$sig"
  cleanup_dispatcher "$DISPATCH_PID"
  # Re-emit the signal-default behavior: non-zero exit.
  exit 1
}
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP
trap 'on_signal INT' INT

on_exit() {
  local code=$?
  # Best-effort PID-file removal; never fail in the EXIT trap.
  if [[ -f "$WATCHDOG_PID_FILE" ]]; then
    local pid_in_file
    pid_in_file="$(cat "$WATCHDOG_PID_FILE" 2>/dev/null || echo '')"
    if [[ "$pid_in_file" == "$$" ]]; then
      rm -f "$WATCHDOG_PID_FILE" 2>/dev/null || true
    fi
  fi
  log_w "EXIT trap: exit_code=$code reason=$EXIT_REASON" 2>/dev/null || true
}
trap on_exit EXIT

# --- single-instance guard ------------------------------------------------

if [[ -f "$WATCHDOG_PID_FILE" ]]; then
  existing_pid="$(cat "$WATCHDOG_PID_FILE" 2>/dev/null || echo '')"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    log_w "refusing to start: another watchdog is alive at pid=$existing_pid (pid file: $WATCHDOG_PID_FILE)"
    EXIT_REASON="already-running"
    exit 2
  fi
  log_w "stale pid file at $WATCHDOG_PID_FILE (pid=$existing_pid not alive); removing"
  rm -f "$WATCHDOG_PID_FILE"
fi
mkdir -p "$(dirname "$WATCHDOG_PID_FILE")" 2>/dev/null || true
echo "$$" > "$WATCHDOG_PID_FILE"
log_w "wrote pid file $WATCHDOG_PID_FILE pid=$$"

# --- helpers --------------------------------------------------------------

count_complete_cells() {
  # Count cells whose metrics.json contains "persona_panel_scores" (the
  # eval-completion sentinel introduced in round-6). Used as the
  # "are we making progress" indicator between dispatcher respawns.
  #
  # Round-7: pipefail-safe. `find -print0 | xargs -0 grep -l` returns 0
  # files cleanly on zero matches; wc -l of empty input is 0.
  if [[ ! -d "$SLAB_ROOT" ]]; then
    echo 0
    return
  fi
  # `find ... -print0` succeeds even with zero matches. `xargs --no-run-if-empty`
  # avoids invoking grep on empty input. `grep -l` writes filenames of matches
  # to stdout; `wc -l` counts them. Each pipe stage handles empty input cleanly.
  local count
  count=$(
    find "$SLAB_ROOT" -type f -name 'metrics.json' -print0 2>/dev/null \
      | xargs -0 --no-run-if-empty grep -l 'persona_panel_scores' 2>/dev/null \
      | wc -l
  )
  # wc -l can emit leading whitespace on some platforms; strip it.
  printf '%s\n' "${count// /}"
}

now_unix() { date +%s; }

log_mtime() {
  if [[ -f "$LOG_FILE" ]]; then
    stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

# --- one dispatcher cycle (writes globals; does NOT run in subshell) ------

# Output globals, set by run_one_dispatcher_cycle:
CYCLE_RC=0
CYCLE_BEFORE=0
CYCLE_AFTER=0

run_one_dispatcher_cycle() {
  local respawn_idx="$1"
  log_w "spawning dispatcher (respawn $respawn_idx/$MAX_RESPAWNS)"
  # Run the dispatcher in this watchdog's process group. The watchdog
  # itself is detached from the SSH session by the caller's `setsid`.
  # The `bash -c "$DISPATCH_CMD"` wrapper exists so DISPATCH_CMD can be
  # a free-form string with redirections / pipes if needed; $! captures
  # the wrapper PID, and `kill -0` / `kill -TERM` on it propagate to the
  # python child via process-group semantics.
  bash -c "$DISPATCH_CMD" >> "$LOG_FILE" 2>&1 &
  local dispatch_pid=$!
  DISPATCH_PID="$dispatch_pid"
  log_w "dispatcher started: pid=$dispatch_pid log=$LOG_FILE"

  CYCLE_BEFORE=$(count_complete_cells)
  log_w "cycle start: complete-cells=$CYCLE_BEFORE"

  # Anchor stall detection to "the later of dispatcher-start-time and
  # last log mtime". This prevents the dispatcher-log-doesn't-exist-yet
  # edge case (where stat returns 0 and `gap = now - 0` is astronomical)
  # from firing a false-positive stall on dispatcher startup.
  local cycle_start
  cycle_start=$(now_unix)

  while kill -0 "$dispatch_pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    # Re-check liveness after the sleep: dispatcher may have exited during it.
    if ! kill -0 "$dispatch_pid" 2>/dev/null; then break; fi
    local mtime
    mtime=$(log_mtime)
    local anchor
    if (( mtime > cycle_start )); then
      anchor=$mtime
    else
      anchor=$cycle_start
    fi
    local gap=$(( $(now_unix) - anchor ))
    local current_count
    current_count=$(count_complete_cells)
    log_w "alive — dispatcher-pid=$dispatch_pid respawn=$respawn_idx/$MAX_RESPAWNS complete-cells=$current_count gap=${gap}s"
    if (( gap > STALL_GAP_SECONDS )); then
      log_w "log stall: ${gap}s > ${STALL_GAP_SECONDS}s; killing pid=$dispatch_pid"
      cleanup_dispatcher "$dispatch_pid"
      break
    fi
  done

  # wait may fail with 127 if the child is already reaped — that's fine.
  CYCLE_RC=0
  wait "$dispatch_pid" 2>/dev/null || CYCLE_RC=$?
  DISPATCH_PID=""
  CYCLE_AFTER=$(count_complete_cells)
  log_w "cycle done: rc=$CYCLE_RC complete-cells $CYCLE_BEFORE -> $CYCLE_AFTER"
}

# --- main loop ------------------------------------------------------------

log_w "started: dispatcher-log=$LOG_FILE slab=$SLAB_ROOT max-respawns=$MAX_RESPAWNS poll=${POLL_SECONDS}s stall-gap=${STALL_GAP_SECONDS}s"
total_complete=$(count_complete_cells)
log_w "initial complete-cell count: $total_complete"

for respawn in $(seq 1 "$MAX_RESPAWNS"); do
  run_one_dispatcher_cycle "$respawn"
  rc=$CYCLE_RC
  before=$CYCLE_BEFORE
  after=$CYCLE_AFTER

  if (( rc == 0 )); then
    log_w "dispatcher exited 0 cleanly; respawn $respawn final, complete=$after"
    EXIT_REASON="dispatcher-success"
    exit 0
  fi

  if (( after <= before )); then
    log_w "NO FORWARD PROGRESS in respawn $respawn (complete $before -> $after); aborting"
    EXIT_REASON="no-progress"
    exit 1
  fi

  log_w "dispatcher exited rc=$rc with progress ($before -> $after); cooling down ${COOL_DOWN_SECONDS}s before respawn"
  sleep "$COOL_DOWN_SECONDS"
done

log_w "hit MAX_RESPAWNS=$MAX_RESPAWNS; exiting"
EXIT_REASON="max-respawns"
exit 1
