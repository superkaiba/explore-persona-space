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
# Pod name is DISCOVERED, never pinned. This round already pivoted once
# (pod-1739-r2v2fit -> pod-1739-r2v2fitg on the cpu-bigmem -> lora-7b venue change),
# and a pinned name substring-matched the successor: liveness kept working by luck
# while the escalation text named a pod that no longer existed and suggested a
# --name-suffix that would not resolve. Discover, and carry the real suffix.
POD_PREFIX=pod-${ISSUE}
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
# MULTI-POD by design: the widened legs shard one pod per behavior, and
# fair-roster v2 runs concurrently on its own. Tracking only the first match
# would leave every sibling pod unwatched — the exact billing hazard this exists
# for, multiplied. Collect them all; the ceiling keys on the OLDEST.
RUNNING=""   # "name<TAB>podid" per line
if [ "$pod_rc" -ne 0 ]; then
  pod_state=unknown
else
  RUNNING="$(printf '%s\n' "$pod_out" \
    | awk -v p="^${POD_PREFIX}" '$1 ~ p && $3 == "running" {print $1"\t"$NF}')"
  if [ -n "$RUNNING" ]; then
    pod_state=running
  elif printf '%s\n' "$pod_out" | awk -v p="^${POD_PREFIX}-" '$1 ~ p {found=1} END{exit !found}'; then
    pod_state=present-not-running
  else
    pod_state=absent
  fi
fi
n_running="$(printf '%s' "$RUNNING" | grep -c . 2>/dev/null || echo 0)"
POD_LIST="$(printf '%s' "$RUNNING" | cut -f1 | paste -sd, - 2>/dev/null)"
[ -z "$POD_LIST" ] && POD_LIST="(none)"

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
# Keyed on POD_ID, not mere presence: a venue pivot replaces the pod, and a
# presence-only clock would keep counting across the replacement and report the
# successor's age as the predecessor's.
# Per-pod first-seen ledger ("podid epoch" per line). Keyed on POD ID so a venue
# pivot resets that pod's clock instead of inheriting its predecessor's age, and
# so a pod finishing does not reset its still-running siblings.
run_h=-1; OLDEST_POD=""; OLDEST_SUFFIX=""
if [ "$pod_state" = "running" ]; then
  now_s="$(date +%s)"
  tmp="${SEEN}.tmp.$$"
  : > "$tmp"
  while IFS="$(printf '\t')" read -r pname pid_; do
    [ -z "$pid_" ] && continue
    seen_at="$(awk -v id="$pid_" '$1==id{print $2; exit}' "$SEEN" 2>/dev/null)"
    case "$seen_at" in ''|*[!0-9]*) seen_at="$now_s" ;; esac
    echo "$pid_ $seen_at" >> "$tmp"
    age=$(( (now_s - seen_at) / 3600 ))
    if [ "$age" -gt "$run_h" ]; then
      run_h="$age"; OLDEST_POD="$pname"
      OLDEST_SUFFIX="${pname#"${POD_PREFIX}-"}"
      [ "$OLDEST_SUFFIX" = "$pname" ] && OLDEST_SUFFIX=""
    fi
  done <<EOF
$RUNNING
EOF
  mv -f "$tmp" "$SEEN"      # prunes pods that are no longer running
else
  rm -f "$SEEN"
fi

echo "$(ts) #${ISSUE} pods=${POD_LIST} n_running=${n_running} state=${pod_state} keep_running=${tag_state} marker_age=${age_min}m oldest_runtime=${run_h}h" >> "$LOG"

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
  msg="EPS #${ISSUE}: pod ${OLDEST_POD} has been RUNNING ${run_h}h (ceiling ${MAX_RUN_H}h); ${n_running} pod(s) live: ${POD_LIST}. keep-running is ${tag_state}, so auto-stop is disabled. Verify progress, then terminate that pod if done or dead: pod.py terminate --issue ${ISSUE} --name-suffix ${OLDEST_SUFFIX} --yes   (the tag is ISSUE-WIDE — remove it only when the LAST pod is done)"
elif [ "$pod_state" = "running" ] && [ "$age_min" -ge "$STALL_MIN" ] 2>/dev/null; then
  # The hazard this watch exists for: billing with no observable progress.
  episode="wedged-owner-${STALL_MIN}"
  msg="EPS #${ISSUE}: ${n_running} pod(s) RUNNING (${POD_LIST}) but no marker for ${age_min}m (keep_running=${tag_state}). Owner may be dead — pods are billing and the tag disables auto-stop."
elif [ "$pod_state" = "absent" ] && [ "$tag_state" = "set" ]; then
  # Pod died/terminated but the shield was never cleared — auto-stop stays disabled
  # for any later pod on this issue.
  episode="tag-orphaned"
  msg="EPS #${ISSUE}: no pods running but keep-running is still SET. Remove the tag so pod-safety auto-stop re-arms: task.py remove-tag ${ISSUE} keep-running"
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
