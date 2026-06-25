#!/usr/bin/env bash
# Leakage-theory program (#660) background orchestrator.
#
#   GATE on #663 (the batch-judge client fix) -> then dispatch the phase chain via
#   /issue --auto, advancing each phase on a clean awaiting_promotion/completed.
#   Halts + surfaces on any block/stall instead of barreling ahead.
#
# Attach to watch live:   tmux attach -t eps-program   (Ctrl-b d to detach)
# Stop it:                touch .claude/cache/program_orchestrator.STOP
#
# Heavy per-phase work is the proven /issue --auto sessions (crash-recovered by the
# autonomous-session watcher); this driver only gates, sequences, and surfaces.
# Promotion stays user-only — phases park at awaiting_promotion; the chain advances on
# the critic-gated PASS, NOT on promotion (per plan §10 autocontinue).

set -uo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:$PATH"
REPO=/home/thomasjiralerspong/explore-persona-space
cd "$REPO" || exit 2
LOG="$REPO/.claude/cache/program_orchestrator.log"
STOP="$REPO/.claude/cache/program_orchestrator.STOP"
POLL=120                 # seconds between status polls
MAX_WAIT_H=48            # per-phase max wait before declaring a stall

log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

status(){ # $1=task id -> prints status ('' on read error)
  uv run python scripts/task.py view "$1" --json 2>/dev/null \
    | python3 -c "import sys,json
try: print(json.load(sys.stdin).get('status',''))
except Exception: print('')" 2>/dev/null
}

wait_terminal(){ # $1=id $2=label ; 0 on completed/awaiting_promotion, 1 on block/stall
  local id="$1" label="$2" s deadline; deadline=$(( $(date +%s) + MAX_WAIT_H*3600 ))
  while true; do
    [ -f "$STOP" ] && { log "STOP sentinel present -> halting."; exit 3; }
    s="$(status "$id")"
    log "  [$label #$id] status=${s:-<read-failed>}"
    case "$s" in
      completed|awaiting_promotion) log "  [$label #$id] DONE ($s)"; return 0;;
      blocked|archived)             log "  [$label #$id] HALT ($s) -> stopping chain, surfacing."; return 1;;
    esac
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log "  [$label #$id] STALL: no terminal state in ${MAX_WAIT_H}h -> stopping, surfacing."; return 1
    fi
    sleep "$POLL"
  done
}

ensure_spawned(){ # $1=id $2=label : spawn /issue --auto unless already active/terminal
  local id="$1" label="$2" s; s="$(status "$id")"
  case "$s" in
    proposed|on_hold|"")
      log "  spawning /issue --auto $id ($label)"
      uv run python scripts/spawn_session.py spawn-issue --issue "$id" --auto 2>&1 | tee -a "$LOG" ;;
    *) log "  $label #$id already at '$s' -> not re-spawning" ;;
  esac
}

log "================ Leakage program (#660) orchestrator START ================"
log "Plan: docs/theory_assumption_test_plan.md | Phases: 1=#658  2=#664  3=#665  4=#666 | Gate=#663"

# ---- GATE: wait for #663 (batch-judge client fix) ----
log "GATE: waiting for #663 (batch-judge client hardening) to finish before any phase dispatches."
if ! wait_terminal 663 "GATE batch-fix"; then
  log "GATE FAILED: #663 did not complete -> NOT dispatching the program. Surfacing."
  touch "$STOP"; exit 1
fi
log "GATE CLEARED: #663 done."

# ---- PHASE 1: #658 (revive the held foundation analysis + the recipe sweep) ----
if [ "$(status 658)" = "on_hold" ]; then
  log "Phase 1: reviving #658 (on_hold -> interpreting)"
  uv run python scripts/task.py set-status 658 interpreting 2>&1 | tee -a "$LOG"
fi
ensure_spawned 658 "Phase 1 foundation"
wait_terminal 658 "Phase 1 (#658)" || { log "Phase 1 halted -> stopping."; touch "$STOP"; exit 1; }

# ---- PHASE 2: #664 (fine-tune fleet) ----
ensure_spawned 664 "Phase 2 fleet"
wait_terminal 664 "Phase 2 (#664)" || { log "Phase 2 halted -> stopping."; touch "$STOP"; exit 1; }

# ---- PHASE 3 + 4 (overlap, §10e) ----
ensure_spawned 665 "Phase 3"
ensure_spawned 666 "Phase 4"
r3=0; r4=0
wait_terminal 665 "Phase 3 (#665)" || r3=1
wait_terminal 666 "Phase 4 (#666)" || r4=1
if [ "$r3" = 0 ] && [ "$r4" = 0 ]; then
  log "================ ALL PHASES reached awaiting_promotion/completed. Program complete (pending your promotions). ================"
else
  log "================ Program finished WITH HALTS: Phase3 rc=$r3  Phase4 rc=$r4 -> surfacing. ================"
  exit 1
fi
