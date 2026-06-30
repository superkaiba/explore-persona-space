#!/bin/bash
# Phase A RESUME — skip the already-done gate_cpu step (24/24 cells have
# g0_E0.json from the previous run that died mid-judge_E); re-run judge_E
# from scratch (the dead driver did NOT persist the 8 in-flight batch IDs,
# so they orphan on the Anthropic side — wf-fix-candidate filed), then run
# the initial aggregate. Same sentinel + log layout as the canonical Phase A.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LOG_DIR="/tmp/issue665-phase-a"
mkdir -p "$LOG_DIR"
SENTINEL_DIR="/workspace/logs"
mkdir -p "$SENTINEL_DIR"

# Source .env so judge_E (Anthropic Batch API) has ANTHROPIC_API_KEY + HF tokens
set -a; [ -f .env ] && source .env; set +a

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*"; }

log "Phase A RESUME start (gate_cpu skipped — 24/24 g0_E0.json present)"

log "Step 2/3: judge_E --scope content (Anthropic Batch API, Sonnet 4.5)"
uv run python scripts/issue665_judge_E.py --scope content 2>&1 | tee "$LOG_DIR/judge_E.log"
log "Step 2/3 done"

log "Step 3/3: aggregate (initial pass — A3.6c arms will fold in later)"
uv run python scripts/issue665_aggregate.py 2>&1 | tee "$LOG_DIR/aggregate_phase_a.log"
log "Step 3/3 done"

# Write the §6.5 completion sentinel (poll_pipeline picks this up)
cat > "$SENTINEL_DIR/issue-665-phase-a-results.json" <<JSON
{
  "phase": "A",
  "completed_at": "$(ts)",
  "steps_done": ["gate_cpu (prior run)", "judge_E (resume)", "aggregate_initial"],
  "next_phase": "B (A3.6c GPU pod)",
  "eval_paths": [
    "eval_results/issue_665/per_cell/*/gate_arms.json",
    "eval_results/issue_665/per_cell/*/judged_E.json",
    "eval_results/issue_665/per_cell/*/g0_E0.json",
    "eval_results/issue_665/aggregate.json"
  ]
}
JSON
log "Phase A RESUME complete; sentinel written to $SENTINEL_DIR/issue-665-phase-a-results.json"
