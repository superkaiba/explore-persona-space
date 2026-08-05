#!/usr/bin/env bash
# 429/529 auto-retry hook — limiter-aware, storm-capped (#1448; 529 added 2026-08-05).
#
# Repo mirror = source of truth. Installed to ~/.claude/hooks/on-stop-429-retry.sh
# by `cp -p`; tests/test_on_stop_429_retry_hook.py pins mirror <-> installed sync.
#
# StopFailure path: Claude Code pre-filters to matcher="rate_limit". The
# Stop event is intentionally NOT handled — its input includes
# last_assistant_message, which can contain quoted API error JSON whenever
# this feature is being discussed, causing self-triggering recursion.
#
# SubagentStop path: no rate_limit matcher is available (SubagentStop's
# matcher filters by agent type, not error). We detect the error by tailing
# the sub-agent's OWN transcript (agent_transcript_path — never the parent's
# transcript_path) and requiring a structured isApiErrorMessage:true line
# whose content mentions 429 / rate limit / 529 / overloaded. Plain prose
# matches do NOT count (the original false-positive recursion). Note:
# SubagentStop payloads carry the PARENT session's session_id (verified on
# real captures, 2026-07-17), so parallel subagent errors in one session
# share that session's storm budget.
#
# 529 Overloaded (added 2026-08-05). A 529 is SERVER-SIDE capacity, not a
# per-minute token budget, so it gets its own pacing: minute-boundary
# alignment is meaningless for it (nothing replenishes at the boundary).
# Instead the wait grows with the storm counter —
# OVERLOADED_BASE_SECS * 2^(count-1), capped at OVERLOADED_MAX_SECS, plus the
# same jitter. Worst case 90+10=100s, inside the hook's 120s settings timeout.
#
# KNOWN RESIDUAL — the StopFailure path is NOT covered for 529. That path is
# gated by the harness's matcher="rate_limit" (~/.claude/settings.json), and a
# 529 Overloaded is not a rate limit, so a main-loop 529 most likely never
# reaches this hook at all. The classification below handles 529 correctly IF
# one ever arrives (strictly an improvement, never a regression), but the gate
# is upstream. Widening it needs a verified matcher name for overloaded /
# api_error — do NOT guess one: a wrong matcher can silently break the working
# 429 wiring. Until then the main-loop 529 backstop remains the watcher's
# prompt-wedge lane (autonomous_session_watch.py — counts isApiErrorMessage
# rows, force-respawns after EPM_TICK_WEDGE_MIN_FAILED_TURNS consecutive
# failed wake turns, default 3). SubagentStop 529s ARE covered here.
#
# Re-wake mechanism (exit 2 + stdout + asyncRewake) is deliberately UNCHANGED.
# Delivery is stderr-first, stdout-fallback ("The hook's stderr, or stdout if
# stderr is empty, is shown to Claude" — code.claude.com/docs/en/hooks), so
# EVERY stderr-capable line below is 2>/dev/null-guarded: any stray stderr on
# the exit-2 path would displace the stdout re-wake message.
#
# Fallback plan (plan §0.0): if re-wake messages stop arriving under this
# flow, shrink the wait and keep the storm counter as the primary defense.
#
# Counter-degrade signature: a persistent count-file write failure means the
# storm cap is silently disabled — the hook degrades to paced-UNBOUNDED
# retries (~40-180/h at 20-89s per wait), never back to the old ~2s storm.
set -u

# Kill switch: touch $DISABLE_FILE to disable the hook entirely (checked
# FIRST, before stdin is read or any file is written).
DISABLE_FILE="${CLAUDE_429_RETRY_DISABLE_FILE:-/tmp/claude-429-retry-disabled}"
[ -f "$DISABLE_FILE" ] && exit 0

STATE_DIR="${CLAUDE_429_RETRY_STATE_DIR:-/tmp/claude-429-retry-state}"
DBG="${CLAUDE_429_RETRY_DEBUG_DIR:-/tmp/claude-stop-hook-debug}"
MAX_CONSECUTIVE="${CLAUDE_429_RETRY_MAX_CONSECUTIVE:-5}"
RESET_WINDOW_SECS="${CLAUDE_429_RETRY_RESET_WINDOW:-600}"
MIN_WAIT_SECS=20
JITTER_MAX_SECS=10
# 529-only pacing (server capacity, not a per-minute bucket): exponential in
# the storm counter. 20/40/80/90/90 + jitter[0,10] -> worst 100s, inside the
# 120s hook timeout declared in ~/.claude/settings.json.
OVERLOADED_BASE_SECS=20
OVERLOADED_MAX_SECS=90
NO_SLEEP="${CLAUDE_429_RETRY_NO_SLEEP:-0}"  # test knob: skip the sleep, still report the wait
FAKE_NOW="${CLAUDE_429_RETRY_FAKE_NOW:-}"   # test knob: overrides now=$(date +%s) when set
                                            # (boundary tests only — counter mtime math also
                                            # reads $now, so counter tests use real time)

# Sanitize every numeric knob to digits-only so no later arithmetic /
# numeric-test can error to stderr (pure-bash pattern match — no stderr).
case "$MAX_CONSECUTIVE" in '' | *[!0-9]*) MAX_CONSECUTIVE=5 ;; esac
case "$RESET_WINDOW_SECS" in '' | *[!0-9]*) RESET_WINDOW_SECS=600 ;; esac
case "$FAKE_NOW" in *[!0-9]*) FAKE_NOW="" ;; esac

input=$(cat)

# Debug capture (preserved; last 10 invocations per event, namespaced).
{ mkdir -p "$DBG"; } 2>/dev/null || true
event=$(printf '%s' "$input" | jq -r '.hook_event_name // "unknown"' 2>/dev/null || echo unknown)
# Sanitize the event name (it becomes a filename component) — safe chars only.
event=$(printf '%s' "$event" | tr -cd 'A-Za-z0-9._-')
[ -z "$event" ] && event=unknown
ts=$(date +%s%N 2>/dev/null || echo 0)
{ printf '%s\n' "$input" > "$DBG/${event}-${ts}.json"; } 2>/dev/null || true
{ ls -t "$DBG"/"${event}"-*.json | tail -n +11 | xargs -r rm -f; } 2>/dev/null || true

case "$event" in
  StopFailure)
    # Matcher pre-filtered to rate_limit — the 429 text (incl. the limiter
    # phrase) lives in the payload's last_assistant_message.
    err_text=$(printf '%s' "$input" | jq -r '.last_assistant_message // ""' 2>/dev/null || echo "")
    ;;
  SubagentStop)
    transcript=$(printf '%s' "$input" | jq -r '.agent_transcript_path // empty' 2>/dev/null || echo "")
    { [ -z "$transcript" ] || [ ! -f "$transcript" ]; } && exit 0
    # Preserved defense: require a structured isApiErrorMessage:true line on a
    # recent transcript line whose content mentions 429 / rate limit / 529 /
    # overloaded. The error test stays INSIDE the jq select (never a post-hoc
    # grep over fallback text) so payload prose can never satisfy the gate; the
    # matched line's content doubles as the limiter-classification text.
    err_text=$(tail -n 8 "$transcript" 2>/dev/null | jq -r -s \
      '[.[] | select(.isApiErrorMessage == true) | (.message.content // "" | tostring)
        | select(test("429|rate.limit|overloaded|(^|[^0-9])529([^0-9.,]|$)"; "i"))] | last // ""' \
      2>/dev/null || echo "")
    [ -z "$err_text" ] && exit 0
    ;;
  *)
    # Any other event (Stop, etc.) is a no-op — see header comment.
    exit 0
    ;;
esac

# Standard loop guard (preserved, defensive).
active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false' 2>/dev/null || echo false)
[ "$active" = "true" ] && exit 0

# --- Error classification (from the 429/529 text) ------------------------
# Anthropic exposes three per-minute dimensions (ITPM / OTPM / RPM); the 429
# body names which was hit. Order matters only for determinism — the three
# phrases are mutually exclusive substrings.
#
# 529 is checked FIRST and is a DIFFERENT class, not a fourth limiter: it is
# server capacity, so it takes the exponential pacing below instead of
# minute-boundary alignment. Match on the word "overloaded" (Anthropic's 529
# carries type "overloaded_error" / message "Overloaded") or on a 529 that is
# code-shaped. The digit guards are load-bearing: a bare `529` would match the
# token counts inside a genuine 429 body (e.g. "529,000 input tokens per
# minute") and mis-pace a rate limit as an overload.
limiter=unknown
if printf '%s' "$err_text" | grep -qiE 'overloaded|(^|[^0-9])529([^0-9.,]|$)'; then
  limiter=overloaded
elif printf '%s' "$err_text" | grep -qi 'output tokens per minute'; then
  limiter=output-TPM
elif printf '%s' "$err_text" | grep -qi 'input tokens per minute'; then
  limiter=input-TPM
elif printf '%s' "$err_text" | grep -qi 'requests per minute'; then
  limiter=RPM
fi

# --- Per-session storm counter (increment BEFORE the cap check) ----------
# Key: sanitized session_id — the PARENT session id on both event paths, so
# a session's parallel subagent 429s share one budget. mtime = arrival time
# of the latest invocation (refreshed on every invocation, including at-cap
# ones); a gap > RESET_WINDOW_SECS starts a fresh storm budget.
sid=$(printf '%s' "$input" | jq -r '.session_id // "unknown"' 2>/dev/null | tr -cd 'A-Za-z0-9._-')
[ -z "$sid" ] && sid=unknown
{ mkdir -p "$STATE_DIR"; } 2>/dev/null || true
cfile="$STATE_DIR/${sid}.count"
now=$(date +%s 2>/dev/null || echo 0)
case "$now" in '' | *[!0-9]*) now=0 ;; esac
[ -n "$FAKE_NOW" ] && now="$FAKE_NOW"
count=0
if [ -f "$cfile" ]; then
  # GNU stat (-c %Y); this hook targets the Linux VM only.
  mtime=$({ stat -c %Y "$cfile"; } 2>/dev/null || echo 0)
  case "$mtime" in '' | *[!0-9]*) mtime=0 ;; esac
  if [ $((now - mtime)) -le "$RESET_WINDOW_SECS" ]; then
    count=$({ tr -cd '0-9' <"$cfile"; } 2>/dev/null || echo "")
    count=${count:-0}
  fi
fi
count=$((count + 1))
{ printf '%s' "$count" > "$cfile"; } 2>/dev/null || true
{ find "$STATE_DIR" -name '*.count' -mmin +60 -delete; } 2>/dev/null || true
# Visibility sentinel (preserved) — purely for "did it fire" checks.
{ touch /tmp/claude-429-retry-cooldown; } 2>/dev/null || true

if [ "$count" -gt "$MAX_CONSECUTIVE" ]; then
  # Storm cap: stay stopped SILENTLY (exit 0, empty stdout). Recovery
  # backstops: the watcher's prompt-wedge / dead-wake lanes and the 45-min
  # /issue-tick cron; a >RESET_WINDOW_SECS quiet gap re-arms a fresh budget.
  exit 0
fi

# --- Pacing ---------------------------------------------------------------
# 429 — next minute boundary (floor MIN_WAIT_SECS) + jitter. Per-minute budgets
# replenish continuously (token bucket), so the binding property is "wait tens
# of seconds"; boundary alignment per the task Goal, jitter to decorrelate a
# fleet-wide herd. Range: [20, 89]s < 120s timeout.
#
# 529 — nothing replenishes at a minute boundary, so boundary alignment is
# meaningless. Back off exponentially in the storm counter instead:
# 20/40/80/90/90 + jitter[0,10] -> worst 100s, still inside the 120s timeout.
if [ "$limiter" = overloaded ]; then
  wait_base="$OVERLOADED_BASE_SECS"
  n=1
  while [ "$n" -lt "$count" ]; do
    wait_base=$((wait_base * 2))
    if [ "$wait_base" -ge "$OVERLOADED_MAX_SECS" ]; then
      wait_base="$OVERLOADED_MAX_SECS"
      break
    fi
    n=$((n + 1))
  done
  wait_secs=$((wait_base + RANDOM % (JITTER_MAX_SECS + 1)))
else
  secs_to_boundary=$((60 - now % 60))
  [ "$secs_to_boundary" -lt "$MIN_WAIT_SECS" ] && secs_to_boundary=$((secs_to_boundary + 60))
  wait_secs=$((secs_to_boundary + RANDOM % (JITTER_MAX_SECS + 1)))
fi
[ "$NO_SLEEP" = "1" ] || sleep "$wait_secs" 2>/dev/null || true

# --- Limiter-accurate re-wake message (stdout; exit 2 + asyncRewake) ------
case "$limiter" in
  output-TPM)
    lim_desc="org-wide output-tokens-per-minute (OTPM)"
    hint="keep responses and tool outputs lean for the next few turns"
    ;;
  input-TPM)
    lim_desc="org-wide input-tokens-per-minute (ITPM)"
    hint="keep prompts lean and stagger any subagent spawns"
    ;;
  RPM)
    lim_desc="org-wide requests-per-minute (RPM)"
    hint="reduce parallel API calls for the next few turns"
    ;;
  overloaded)
    lim_desc="529 Overloaded — upstream server capacity, NOT your token budget"
    hint="retry as-is; thinning prompts or reducing parallelism does not help an overload"
    ;;
  *)
    lim_desc="rate limit (class unknown)"
    hint="pace token usage for the next few turns"
    ;;
esac
if [ "$event" = "SubagentStop" ]; then
  # .agent_type can be empty string (truthy in jq's `//`), so post-process.
  agent=$(printf '%s' "$input" | jq -r '.agent_type // empty' 2>/dev/null || echo "")
  [ -z "$agent" ] && agent=$(printf '%s' "$input" | jq -r '.agent_id // empty' 2>/dev/null || echo "")
  [ -z "$agent" ] && agent="the sub-agent"
  if [ "$limiter" = overloaded ]; then
    printf 'A sub-agent (%s) was blocked by %s. This hook already waited %ss (exponential backoff + jitter; retry %s/%s this storm). Re-spawn the same sub-agent with the same prompt and continue; %s.\n' \
      "$agent" "$lim_desc" "$wait_secs" "$count" "$MAX_CONSECUTIVE" "$hint"
  else
    printf 'A sub-agent (%s) was blocked by a 429 on the %s limit. This hook already waited %ss (to the minute boundary + jitter; retry %s/%s this storm). Re-spawn the same sub-agent with the same prompt and continue; %s.\n' \
      "$agent" "$lim_desc" "$wait_secs" "$count" "$MAX_CONSECUTIVE" "$hint"
  fi
else
  if [ "$limiter" = overloaded ]; then
    printf 'This session hit %s. This hook already waited %ss (exponential backoff + jitter; retry %s/%s this storm). Continue your prior task from where you left off; %s.\n' \
      "$lim_desc" "$wait_secs" "$count" "$MAX_CONSECUTIVE" "$hint"
  else
    printf 'This session hit the %s limit (429). This hook already waited %ss (to the minute boundary + jitter; retry %s/%s this storm). Continue your prior task from where you left off; %s.\n' \
      "$lim_desc" "$wait_secs" "$count" "$MAX_CONSECUTIVE" "$hint"
  fi
fi
exit 2
