#!/usr/bin/env bash
# Pipeline driver for issue #498 — trait role-header vs system-prompt.
# Plan v1.2 §4.1 phases 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

echo "[phase=preflight] $(date -Iseconds)"
uv run python scripts/i498_phase0_preflight.py

echo "[phase=codepath_verify] $(date -Iseconds)"
uv run python scripts/i498_phase0_codepath_verify.py

echo "[phase=r_pos] $(date -Iseconds)"
uv run python scripts/i498_phase1_generate_RPos.py

echo "[phase=r_neg] $(date -Iseconds)"
uv run python scripts/i498_phase1_generate_RNeg.py

echo "[phase=phase2_smoke] $(date -Iseconds)"
uv run python scripts/i498_phase23_train.py --arms role --seeds 42 --smoke --gpu-id 0
uv run python scripts/i498_phase2_smoke_judge.py \
    --adapter adapters/i498_role_seed42_smoke --arm role --threshold 3.0

echo "[phase=phase3_sweep] $(date -Iseconds)"
# Wave 1: 4 cells in parallel on 4 GPUs.
(uv run python scripts/i498_phase23_train.py --arms system --seeds 42 --gpu-id 0 > "$LOG_DIR/i498_system_seed42.log" 2>&1) &
(uv run python scripts/i498_phase23_train.py --arms system --seeds 137 --gpu-id 1 > "$LOG_DIR/i498_system_seed137.log" 2>&1) &
(uv run python scripts/i498_phase23_train.py --arms role --seeds 42 --gpu-id 2 > "$LOG_DIR/i498_role_seed42.log" 2>&1) &
(uv run python scripts/i498_phase23_train.py --arms role --seeds 137 --gpu-id 3 > "$LOG_DIR/i498_role_seed137.log" 2>&1) &
wait
# Wave 2: 2 cells.
(uv run python scripts/i498_phase23_train.py --arms system --seeds 1337 --gpu-id 0 > "$LOG_DIR/i498_system_seed1337.log" 2>&1) &
(uv run python scripts/i498_phase23_train.py --arms role --seeds 1337 --gpu-id 1 > "$LOG_DIR/i498_role_seed1337.log" 2>&1) &
wait

echo "[phase=phase4_eval] $(date -Iseconds)"
uv run python scripts/i498_phase4_eval.py

echo "[phase=phase4_judge] $(date -Iseconds)"
uv run python scripts/i498_phase4_judge.py

echo "[phase=phase5_analyze] $(date -Iseconds)"
uv run python scripts/i498_phase5_analyze.py
uv run python scripts/plot_i498_clean_result.py

# End-of-run sentinel for poll_pipeline.py.
SENTINEL="$LOG_DIR/issue-498-epm_results-$(date +%s).json"
cat > "$SENTINEL" <<EOF
{
  "sentinel_schema_version": 1,
  "kind": "epm:results",
  "version": 1,
  "task_id": "498",
  "by": "i498_run_all.sh",
  "ts": "$(date -Iseconds)",
  "note": "Pipeline completed end-to-end; see eval_results/issue_498/analysis.json + figures/issue_498/."
}
EOF
echo "Sentinel: $SENTINEL"

echo "[phase=done] $(date -Iseconds)"
