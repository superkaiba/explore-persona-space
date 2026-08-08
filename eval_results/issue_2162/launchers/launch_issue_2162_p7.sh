#!/bin/bash
# P7 analysis chain for issue #2162 — four explicit steps, NOT --step all
# (step_margin asserts on an empty margin dir with no all-carve-out; margin is
# deliberately deferred and stage2 has not happened yet).
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a

STAGE=/workspace/issue2162_stage/issue2162_ctxinfo
FLAGS=(
  --rollouts-dir "$STAGE/raw_completions/grid"
  --anchors-dir  "$STAGE/raw_completions/anchors"
  --va-dir       "$STAGE/analysis_tensors/va_store"
  --bank-pt      "$STAGE/analysis_tensors/vc_bank/vc_bank.pt"
  --scores-dir   "$STAGE/raw_completions/judge_raw/scores"
  --out-dir      eval_results/issue_2162/f_metrics
)

for step in f-tables stats probe two-by-two; do
  echo "=== STEP $step START $(date -u +%FT%TZ) ==="
  uv run python scripts/issue2162_analysis.py --step "$step" "${FLAGS[@]}"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "=== STEP $step FAILED rc=$rc - chain aborted; terminal phase token suppressed ==="
    exit "$rc"
  fi
  echo "=== STEP $step DONE $(date -u +%FT%TZ) ==="
done

echo "P7 ANALYSIS CHAIN COMPLETE rc=0"
echo "[phase=done]"
