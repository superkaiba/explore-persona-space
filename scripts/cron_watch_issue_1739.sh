#!/usr/bin/env bash
# Durable periodic monitor for task #1739 (r2v2 P-A/P-B fit + extraction factorial).
#
# Survives the chat session that created it. Complements — never replaces — the
# fleet-wide cron_autonomous_session_watch.sh. This script only WATCHES: it logs a
# status line each run and escalates once per episode. It never re-drives /issue,
# never mutates task state, and never stops or terminates a pod.
#
# WHY THIS EXISTS, and why it is not a copy of cron_watch_issue_2091.sh:
# #1739 already sits at `awaiting_promotion` — a status that script treats as
# TERMINAL and self-retires on. Here that status is not terminal in substance: a
# suffixed follow-up pod (pod-1739-r2v2fit) is running a multi-hour fit under a
# `keep-running` tag, driven by a subagent that dies with its chat session.
#
# That combination is the real hazard this watch exists for (#1582 class):
#   pod RUNNING + keep-running SET + owner DEAD  ==  indefinite billing, no auto-stop.
# The tag deliberately disables the watcher's pod-safety auto-stop, so nothing else
# in the fleet will catch it.
#
# Retires when the pod is gone AND the keep-running tag has been removed — i.e. when
# the round has actually been torn down, not when the task status says so.
#
# Remove early:  crontab -l | grep -v cron_watch_issue_1739 | crontab -
set -uo pipefail

ISSUE=1739
POD=pod-1739-r2v2fit
REPO=/home/thomasjiralerspong/explore-persona-space
LOG=/home/thomasjiralerspong/my-goat/logs/watch_issue_${ISSUE}.log
STATE=/home/thomasjiralerspong/.eps-autonomous/watch-${ISSUE}.state
PUSH=/home/thomasjiralerspong/my-goat/scripts/telegram_push.sh
STALL_MIN=${EPS_WATCH_1739_STALL_MIN:-60}
MAX_RUN_H=${EPS_WATCH_1739_MAX_RUN_H:-8}
SEEN=/home/thomasjiralerspong/.eps-autonomous/watch-${ISSUE}.firstseen

cd "$REPO" || exit 0
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# --- pod liveness (live API is authoritative; rc!=0 is UNKNOWN, never "gone") ---
pod_out="$(uv run python scripts/pod.py list-ephemeral --issue "$ISSUE" 2>/dev/null)"
pod_rc=$?
if [ "$pod_rc" -ne 0 ]; then
  pod_state=unknown
elif printf '%s' "$pod_out" | grep -q "$POD.*running"; then
  pod_state=running
elif printf '%s' "$pod_out" | grep -q "$POD"; then
  pod_state="$(printf '%s' "$pod_out" | grep "$POD" | awk '{print $3}' | head -1)"
else
  pod_state=absent
fi

# --- keep-running tag (frontmatter is the source of truth; --json omits tags) ---
tag_state=absent
tpath="$(uv run python scripts/task.py find "$ISSUE" 2>/dev/null)"
if [ -n "$tpath" ] && [ -f "$tpath/body.md" ]; then
  if awk '/^---$/{c++} c==1' "$tpath/body.md" 2>/dev/null | grep -q 'keep-running'; then
    tag_state=set
  fi
fi

# --- marker age in minutes, from the newest event ---
age_min="$(uv run python -c '
import json,subprocess,datetime
try:
    d=json.loads(subprocess.run(["uv","run","python","scripts/task.py","view","'"$ISSUE"'","--json"],
                                capture_output=True,text=True).stdout)
    t=d["events"][-1]["ts"].replace("Z","+00:00")
    now=datetime.datetime.now(datetime.timezone.utc)
    print(int((now-datetime.datetime.fromisoformat(t)).total_seconds()//60))
except Exception: print(-1)' 2>/dev/null)"
[ -z "$age_min" ] && age_min=-1

# --- pod runtime, tracked locally ------------------------------------------
# Deliberately NOT derived from marker age: markers are posted by ANY actor
# (including the chat orchestrator), so a marker clock can look healthy while the
# pod's actual owner is dead. Wall-clock runtime is owner-independent.
run_h=-1
if [ "$pod_state" = "running" ]; then
  [ -f "$SEEN" ] || date +%s > "$SEEN"
  first="$(cat "$SEEN" 2>/dev/null || echo 0)"
  case "$first" in ''|*[!0-9]*) first=0 ;; esac
  [ "$first" -gt 0 ] && run_h=$(( ( $(date +%s) - first ) / 3600 ))
else
  rm -f "$SEEN"
fi

echo "$(ts) #${ISSUE} pod=${pod_state} keep_running=${tag_state} marker_age=${age_min}m pod_runtime=${run_h}h" >> "$LOG"

# --- retire only when the round is genuinely torn down -----------------------
if [ "$pod_state" = "absent" ] && [ "$tag_state" = "absent" ]; then
  echo "$(ts) #${ISSUE} pod gone + tag cleared — round torn down, removing this cron" >> "$LOG"
  [ -x "$PUSH" ] && "$PUSH" "EPS #${ISSUE} r2v2fit round torn down (pod gone, keep-running cleared) — monitor retired." >/dev/null 2>&1
  crontab -l 2>/dev/null | grep -v "cron_watch_issue_${ISSUE}" | crontab - 2>/dev/null
  rm -f "$STATE"
  exit 0
fi

# --- escalate once per episode ----------------------------------------------
episode=""
msg=""
if [ "$pod_state" = "running" ] && [ "$run_h" -ge "$MAX_RUN_H" ] 2>/dev/null; then
  # Owner-independent backstop. Fires even if markers look fresh, because the
  # marker clock is reset by any actor and cannot prove the fit is still alive.
  episode="runtime-${MAX_RUN_H}h"
  msg="EPS #${ISSUE}: pod ${POD} has been RUNNING ${run_h}h (ceiling ${MAX_RUN_H}h). The fit was scoped at ~4h wall. keep-running is ${tag_state}, so auto-stop is disabled. Verify progress, then terminate if done or dead: task.py remove-tag ${ISSUE} keep-running && pod.py terminate --issue ${ISSUE} --name-suffix r2v2fit --yes"
elif [ "$pod_state" = "running" ] && [ "$age_min" -ge "$STALL_MIN" ] 2>/dev/null; then
  # The hazard this watch exists for: billing with no observable progress.
  episode="wedged-owner-${STALL_MIN}"
  msg="EPS #${ISSUE}: pod ${POD} RUNNING but no marker for ${age_min}m (keep_running=${tag_state}). Owner may be dead — pod is billing and the tag disables auto-stop. Check, then terminate if the round is done or dead."
elif [ "$pod_state" = "absent" ] && [ "$tag_state" = "set" ]; then
  # Pod died/terminated but the shield was never cleared — auto-stop stays disabled
  # for any later pod on this issue.
  episode="tag-orphaned"
  msg="EPS #${ISSUE}: pod ${POD} is gone but keep-running is still SET. Remove the tag so pod-safety auto-stop re-arms: task.py remove-tag ${ISSUE} keep-running"
fi

if [ -n "$episode" ]; then
  prev="$(cat "$STATE" 2>/dev/null || true)"
  if [ "$prev" != "$episode" ]; then
    echo "$episode" > "$STATE"
    [ -x "$PUSH" ] && "$PUSH" "$msg" >/dev/null 2>&1
    echo "$(ts) #${ISSUE} ESCALATED (${episode})" >> "$LOG"
  fi
else
  rm -f "$STATE"   # healthy again -> re-arm
fi

exit 0
