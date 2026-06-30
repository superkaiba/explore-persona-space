#!/bin/bash
# Phase A — CPU on VM (no pod, no GPU billing).
# Pipeline: gate_cpu (all CPU arms) → judge_E (Batch API) → initial aggregate.
# Writes per-step + final sentinels to /tmp/issue665-phase-a-*.json.
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

log "Phase A start (CPU arms over content scope = 8 bm + 8 ic + 5 tf = 21 cells)"
log "Step 1/3: gate_cpu --scope content (all 9 CPU arms over 21 cells, all layers)"
uv run python scripts/issue665_gate_cpu.py --scope content 2>&1 | tee "$LOG_DIR/gate_cpu.log"
log "Step 1/3 done"

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
  "steps_done": ["gate_cpu", "judge_E", "aggregate_initial"],
  "next_phase": "B (A3.6c GPU pod)",
  "eval_paths": [
    "eval_results/issue_665/per_cell/*/gate_arms.json",
    "eval_results/issue_665/per_cell/*/judged_E.json",
    "eval_results/issue_665/per_cell/*/g0_E0.json",
    "eval_results/issue_665/aggregate.json"
  ]
}
JSON
log "Phase A complete; sentinel written to $SENTINEL_DIR/issue-665-phase-a-results.json"
