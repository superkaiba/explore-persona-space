#!/usr/bin/env bash
# Issue #489 — top-level pipeline launcher (pod-side).
#
# Plan v5 §4.6. Runs Phase 0 → 0a → 1 → 2 (smoke) → 3 (sweep) → 4 → 5 → figures
# end-to-end. Emits [phase=<name>] lines for poll_pipeline.py and writes an
# end-of-run sentinel for the VM orchestrator.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_489
SENTINEL_DIR=/workspace/logs
mkdir -p "$LOG_DIR" "$SENTINEL_DIR"

SEED=42
SKIP_PHASE0=0
SKIP_PHASE0A=0

for arg in "$@"; do
    case "$arg" in
        --seed=*) SEED="${arg#*=}" ;;
        --skip-phase0) SKIP_PHASE0=1 ;;
        --skip-phase0a) SKIP_PHASE0A=1 ;;
        *) ;;
    esac
done

echo "[phase=preflight] === i489 run_all $(date -Iseconds) seed=$SEED ==="

# Phase 0 — generate R for 24 union contexts over Q_train + Q_test.
if [ "$SKIP_PHASE0" -eq 0 ]; then
    echo "[phase=phase0_generate_data] === Phase 0 data gen $(date -Iseconds) ==="
    uv run python scripts/i489_phase0_generate_data.py --split both \
        > "$LOG_DIR/phase0.log" 2>&1
fi

# Phase 0a — Claude-as-judge SP-string identity check (M2).
if [ "$SKIP_PHASE0A" -eq 0 ]; then
    echo "[phase=phase0a_sp_identity_check] === Phase 0a SP identity check $(date -Iseconds) ==="
    uv run python scripts/i489_phase0_sp_identity_check.py \
        > "$LOG_DIR/phase0a.log" 2>&1
fi

# Phase 1 — predictors + cosine coverage gate.
echo "[phase=phase1_predictors] === Phase 1 predictors $(date -Iseconds) ==="
uv run python scripts/i489_phase1_predictors.py --phase all \
    > "$LOG_DIR/phase1.log" 2>&1

# Phase 2/3 — smoke + sweep via the SAME dispatcher.
echo "[phase=phase23_train] === Phase 2/3 train sweep $(date -Iseconds) ==="
bash scripts/i489_phase23_dispatch.sh --seed="$SEED" \
    > "$LOG_DIR/phase23.log" 2>&1

# Phase 4 — on-policy eval + HF teacher-forced ΔG.
echo "[phase=phase4_eval] === Phase 4 eval $(date -Iseconds) ==="
bash scripts/i489_phase4_dispatch.sh --seed="$SEED" \
    > "$LOG_DIR/phase4.log" 2>&1

# Phase 5 — analyze.
echo "[phase=phase5_analyze] === Phase 5 analyze $(date -Iseconds) ==="
uv run python scripts/i489_phase5_analyze.py --seed "$SEED" \
    > "$LOG_DIR/phase5.log" 2>&1

# Figures.
echo "[phase=make_figures] === Figures $(date -Iseconds) ==="
uv run python scripts/i489_make_figures.py \
    > "$LOG_DIR/figures.log" 2>&1

# End-of-run sentinel for poll_pipeline.py.
epoch="$(date +%s)"
sentinel="${SENTINEL_DIR}/issue-489-epm_results-${epoch}.json"
uv run python - <<EOF
import json, datetime
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "issue": 489,
    "phase": "all_done",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": "All phases (0 -> 0a -> 1 -> 23 -> 4 -> 5 -> figures) completed.",
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote results sentinel: $sentinel")
EOF

echo "[phase=done] === i489 run_all complete $(date -Iseconds) ==="
