#!/usr/bin/env bash
# Durable periodic monitor for task #2091 (deterministic vs stochastic decoding).
#
# Survives the chat session that created it. Complements — never replaces — the
# fleet-wide cron_autonomous_session_watch.sh, which owns crash recovery,
# pod-safety and gate-transition pushes for every registered autonomous session.
# This script only WATCHES: it logs a status line each run and escalates once per
# episode when the task is blocked or its marker clock has stalled. It never
# re-drives /issue, never mutates task state, and never stops a pod.
#
# Self-deletes its own crontab line when #2091 reaches a terminal state.
#
# Remove early:  crontab -l | grep -v cron_watch_issue_2091 | crontab -
set -uo pipefail

ISSUE=2091
REPO=/home/thomasjiralerspong/explore-persona-space
LOG=/home/thomasjiralerspong/my-goat/logs/watch_issue_${ISSUE}.log
STATE=/home/thomasjiralerspong/.eps-autonomous/watch-${ISSUE}.state
PUSH=/home/thomasjiralerspong/my-goat/scripts/telegram_push.sh
STALL_MIN=${EPS_WATCH_STALL_MIN:-180}

cd "$REPO" || exit 0
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

verdict="$(uv run python scripts/tick_triage.py "$ISSUE" 2>&1 | head -1)"
status="$(uv run python scripts/task.py view "$ISSUE" --json 2>/dev/null \
  | uv run python -c 'import json,sys; print(json.load(sys.stdin)["path"].split("/")[1])' 2>/dev/null)"
[ -z "$status" ] && status=unknown

# marker age in minutes, from the newest event
age_min="$(uv run python -c '
import json,subprocess,datetime,sys
try:
    d=json.loads(subprocess.run(["uv","run","python","scripts/task.py","view","'"$ISSUE"'","--json"],
                                capture_output=True,text=True).stdout)
    ts=d["events"][-1]["ts"].replace("Z","+00:00")
    now=datetime.datetime.now(datetime.timezone.utc)
    print(int((now-datetime.datetime.fromisoformat(ts)).total_seconds()//60))
except Exception: print(-1)' 2>/dev/null)"
[ -z "$age_min" ] && age_min=-1

echo "$(ts) #${ISSUE} status=${status} marker_age=${age_min}m :: ${verdict}" >> "$LOG"

# --- terminal: log, unregister, self-remove -------------------------------
case "$status" in
  awaiting_promotion|completed|archived)
    echo "$(ts) #${ISSUE} TERMINAL (${status}) — removing this cron" >> "$LOG"
    [ -x "$PUSH" ] && "$PUSH" "EPS #${ISSUE} reached ${status} — monitor retired." >/dev/null 2>&1
    crontab -l 2>/dev/null | grep -v "cron_watch_issue_${ISSUE}" | crontab - 2>/dev/null
    rm -f "$STATE"
    exit 0
    ;;
esac

# --- escalate once per episode --------------------------------------------
episode=""
if [ "$status" = "blocked" ]; then
  episode="blocked"
elif [ "$age_min" -ge "$STALL_MIN" ] 2>/dev/null; then
  episode="stall-${STALL_MIN}"
fi

if [ -n "$episode" ]; then
  prev="$(cat "$STATE" 2>/dev/null || true)"
  if [ "$prev" != "$episode" ]; then
    echo "$episode" > "$STATE"
    msg="EPS #${ISSUE} needs you: status=${status}, no marker for ${age_min}m. ${verdict}"
    [ -x "$PUSH" ] && "$PUSH" "$msg" >/dev/null 2>&1
    echo "$(ts) #${ISSUE} ESCALATED (${episode})" >> "$LOG"
  fi
else
  rm -f "$STATE"   # healthy again -> re-arm
fi

exit 0
