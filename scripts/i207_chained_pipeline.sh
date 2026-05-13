#!/usr/bin/env bash
# Chained pipeline for issue #343: finish seed=42 (currently in flight),
# then seeds 137, 256, then Stages 4, 5, 6.
#
# Each step is idempotent — if its output exists, it's skipped.
# Output goes to /workspace/logs/i343_chain.log so a follow-up monitor can
# pick up where we left off.

set -uo pipefail

export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; source .env 2>/dev/null; set +a

LOG=/workspace/logs/i343_chain.log
echo "=== Chained pipeline start: $(date) ===" | tee -a "$LOG"

# ── Step A: seeds 137 + 256 (orchestrator handles 4-way parallel per batch) ──
for SEED in 137 256; do
  echo "" | tee -a "$LOG"
  echo "=== Batch seed=$SEED — starting at $(date) ===" | tee -a "$LOG"

  uv run python scripts/run_i207_gentle_orchestrate.py --seeds "$SEED" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "=== Batch seed=$SEED done (rc=$rc) at $(date) ===" | tee -a "$LOG"

  if [[ $rc -ne 0 ]]; then
    echo "Orchestrator returned non-zero for seed=$SEED — continuing anyway (idempotent retry available)" | tee -a "$LOG"
  fi
done

# ── Step B: base-model greedy generations (Stage 4) ──
echo "" | tee -a "$LOG"
echo "=== Stage 4: base-model generations — starting at $(date) ===" | tee -a "$LOG"
uv run python scripts/i207_base_generate.py --gpu 0 2>&1 | tee -a "$LOG"
echo "=== Stage 4 done at $(date) ===" | tee -a "$LOG"

# ── Step C: 40x40 JS divergence matrix (Stage 5) ──
echo "" | tee -a "$LOG"
echo "=== Stage 5: JS divergence matrix — starting at $(date) ===" | tee -a "$LOG"
uv run python scripts/i207_compute_js_matrix.py --gpu 0 2>&1 | tee -a "$LOG"
echo "=== Stage 5 done at $(date) ===" | tee -a "$LOG"

# ── Step D: Regression CSV + OLS (Stages 6-7) ──
echo "" | tee -a "$LOG"
echo "=== Stage 6-7: regression — starting at $(date) ===" | tee -a "$LOG"
uv run python scripts/i207_run_regression.py 2>&1 | tee -a "$LOG"
echo "=== Stage 6-7 done at $(date) ===" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== CHAINED PIPELINE COMPLETE: $(date) ===" | tee -a "$LOG"
