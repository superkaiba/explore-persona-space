#!/bin/bash
# task #742 production pipeline (VM-side, 0-GPU CPU work).
# 5 sequential phases over existing #658 artifacts. Each phase reads the prior
# phase's JSON output from eval_results/issue_742/.
set -euo pipefail

cd /home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-742

run_phase() {
  local k="$1"
  local label="$2"
  shift 2
  echo "[$(date -Iseconds)] phase ${k} START ${label}"
  if ! "$@"; then
    local rc=$?
    echo "[$(date -Iseconds)] FAIL phase=${k} (${label}) rc=${rc}"
    exit "${rc}"
  fi
  echo "[$(date -Iseconds)] phase ${k} DONE ${label}"
}

# Phase 1 — Stage-0a: Anthropic Batch judge re-rerun (threshold_base=0 forces Batch route), J=20, R=2
run_phase 1 judge_rerun        uv run python scripts/issue742_judge_rerun.py
# Phase 2 — Stage-0b: split-half + binomial sqrt(r_yy); cluster-bootstrap bracket, B=2000, seed=742
run_phase 2 reliability        uv run python scripts/issue742_reliability.py
# Phase 3 — Stage-1: LEACE + dCor permutation (gated internally on Stage-0 bracket headroom)
run_phase 3 nonlinear_residual uv run python scripts/issue742_nonlinear_residual.py
# Phase 4 — Stage-2: subsample learning curve n'=10..50, bootstrap, extrapolate n-to-resolve
run_phase 4 learning_curve     uv run python scripts/issue742_learning_curve.py
# Phase 5 — aggregate figures
run_phase 5 figures            uv run python scripts/issue742_figures.py

echo "[$(date -Iseconds)] OK pipeline complete"
