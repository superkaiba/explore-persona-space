#!/bin/bash
# task #742 production pipeline (VM-side, 0-GPU CPU work).
# 5 sequential phases over existing #658 artifacts. Each phase reads the prior
# phase's JSON output from eval_results/issue_742/.
set -euo pipefail

# Route HF caches to the /mnt/eps-data data disk (503G free, attached 2026-06-30
# per the auto-memory note) so any `hf_hub_download` from `snapshot_raw_completions`
# (and any other phase that touches HF) does NOT refill `/`. The previous launch
# hit OSError(errno=28) at ~8h after `~/.cache/huggingface` swelled to 199G on
# the 485G VM root disk.
export HF_HOME=/mnt/eps-data/thomasjiralerspong/huggingface-cache
mkdir -p "$HF_HOME"

cd /home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-742

run_phase() {
  local k="$1"
  local label="$2"
  shift 2
  echo "[$(date -Iseconds)] phase ${k} START ${label}"
  # POSITIVE-form test so `$?` captures the failing command's rc.  The earlier
  # `if ! "$@"; then rc=$?` inverted the check, so `$?` was always 0 when the
  # FAIL branch fired (the `!` expression's rc).  Inverted -> the FAIL line on
  # disk-full read `FAIL phase=1 rc=0`, masking the real OSError.
  if "$@"; then
    echo "[$(date -Iseconds)] phase ${k} DONE ${label}"
  else
    local rc=$?
    echo "[$(date -Iseconds)] FAIL phase=${k} (${label}) rc=${rc}"
    exit "${rc}"
  fi
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
