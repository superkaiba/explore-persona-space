#!/usr/bin/env bash
# Round-6 (issue #365) external watchdog for the dispatcher.
#
# Survives dispatcher death by living in a separate process group (setsid)
# and respawning the dispatcher up to MAX_RESPAWNS times, with a sanity
# check that progress is being made between respawns (delta of cells with
# a non-empty metrics.json containing persona_panel_scores).
#
# Usage (from inside /workspace/explore-persona-space on the pod):
#
#   setsid bash scripts/watchdog_factor_screen_365.sh \
#       /workspace/logs/issue-365-r6.log \
#       eval_results/issue_365 \
#       "uv run python scripts/dispatch_factor_screen_365.py \
#           --slab-root eval_results/issue_365 \
#           --pool-dir data/issue_365/pools \
#           --sources librarian,surgeon,programmer \
#           --seeds 42 \
#           --skip-pool-stage" \
#       < /dev/null > /workspace/logs/issue-365-watchdog.log 2>&1 &
#   disown
#
# Exits 0 when the dispatcher exits 0. Exits 1 if MAX_RESPAWNS hit without
# forward progress. Tail the watchdog log + the dispatcher log to track.

set -u

LOG_FILE="${1:?usage: $0 LOG_FILE SLAB_ROOT DISPATCH_CMD}"
SLAB_ROOT="${2:?usage: $0 LOG_FILE SLAB_ROOT DISPATCH_CMD}"
DISPATCH_CMD="${3:?usage: $0 LOG_FILE SLAB_ROOT DISPATCH_CMD}"

MAX_RESPAWNS="${WATCHDOG_MAX_RESPAWNS:-5}"
STALL_GAP_SECONDS="${WATCHDOG_STALL_GAP_SECONDS:-1800}"  # 30 min log mtime gap
POLL_SECONDS="${WATCHDOG_POLL_SECONDS:-300}"             # 5 min poll interval

count_complete_cells() {
  # Count cells whose metrics.json contains "persona_panel_scores" (the
  # eval-completion sentinel introduced in round-6). Used as the
  # "are we making progress" indicator between dispatcher respawns.
  if [[ ! -d "$SLAB_ROOT" ]]; then
    echo 0
    return
  fi
  grep -lr "persona_panel_scores" "$SLAB_ROOT" 2>/dev/null | grep -c "metrics.json" || echo 0
}

now_unix() { date +%s; }

log_mtime() {
  if [[ -f "$LOG_FILE" ]]; then
    stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

run_one_dispatcher_cycle() {
  local respawn_idx="$1"
  echo "[watchdog $(date -u +%H:%M:%S)] Spawning dispatcher (respawn $respawn_idx/$MAX_RESPAWNS)" \
      >> "$LOG_FILE"
  # Run the dispatcher in this watchdog's process group. The watchdog
  # itself is detached from the SSH session by the caller's `setsid`.
  bash -c "$DISPATCH_CMD" >> "$LOG_FILE" 2>&1 &
  local dispatch_pid=$!
  echo "[watchdog] dispatch pid=$dispatch_pid" >> "$LOG_FILE"
  local before_count
  before_count=$(count_complete_cells)
  while kill -0 "$dispatch_pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    local gap=$(( $(now_unix) - $(log_mtime) ))
    if (( gap > STALL_GAP_SECONDS )); then
      echo "[watchdog $(date -u +%H:%M:%S)] Log stall: ${gap}s > ${STALL_GAP_SECONDS}s; killing pid=$dispatch_pid" \
          >> "$LOG_FILE"
      kill -TERM "$dispatch_pid" 2>/dev/null || true
      sleep 10
      kill -KILL "$dispatch_pid" 2>/dev/null || true
      break
    fi
  done
  wait "$dispatch_pid" 2>/dev/null
  local rc=$?
  local after_count
  after_count=$(count_complete_cells)
  echo "[watchdog $(date -u +%H:%M:%S)] Cycle done: rc=$rc, complete cells $before_count -> $after_count" \
      >> "$LOG_FILE"
  echo "$rc:$before_count:$after_count"
}

echo "[watchdog $(date -u +%H:%M:%S)] Started. log=$LOG_FILE slab=$SLAB_ROOT max=$MAX_RESPAWNS" \
    >> "$LOG_FILE"
total_complete=$(count_complete_cells)
echo "[watchdog] initial complete-cell count: $total_complete" >> "$LOG_FILE"

for respawn in $(seq 1 "$MAX_RESPAWNS"); do
  result=$(run_one_dispatcher_cycle "$respawn")
  rc=${result%%:*}
  rest=${result#*:}
  before=${rest%%:*}
  after=${rest#*:}

  if (( rc == 0 )); then
    echo "[watchdog $(date -u +%H:%M:%S)] Dispatcher exited 0 cleanly; respawn $respawn final, complete=$after" \
        >> "$LOG_FILE"
    exit 0
  fi

  if (( after <= before )); then
    echo "[watchdog $(date -u +%H:%M:%S)] NO FORWARD PROGRESS in respawn $respawn (complete $before -> $after); aborting" \
        >> "$LOG_FILE"
    exit 1
  fi

  echo "[watchdog $(date -u +%H:%M:%S)] Dispatcher exited rc=$rc with progress ($before -> $after); cooling down 60s" \
      >> "$LOG_FILE"
  sleep 60
done

echo "[watchdog $(date -u +%H:%M:%S)] Hit MAX_RESPAWNS=$MAX_RESPAWNS; exiting" >> "$LOG_FILE"
exit 1
