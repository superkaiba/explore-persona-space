#!/usr/bin/env bash
# Issue #489 — top-level pipeline launcher (pod-side).
#
# Plan v5 §4.6 + B2 + B3 round-2 fixes:
#  - Phase 0a (SP-identity check) runs BEFORE Phase 0 data-gen (B3) so any
#    accepted rewrite freezes into frozen_sp_strings.json BEFORE downstream
#    phases import i489_contexts.
#  - Smoke phase actually invokes scripts/i489_phase2_smoke_calibrate.py (B2),
#    which writes a picked_fracs JSON. Sweep + Phase 4 read that JSON and use
#    the picked fracs, NOT a hardcoded [0.25, 0.50, 1.00].
#
# Round-6 (M2 loosened): Phase 0a NO LONGER blocks the whole run on a
# non-"same" judge verdict. It records per-pair verdicts to
# matched_pair_identity_verdicts.json and exits 0; Phase 5 reads that file
# and scopes the H4(b) confirmatory test to confirmatory pairs only (with a
# graceful UNANSWERED_NO_CONFIRMATORY_PAIRS verdict when none qualify), and
# always reports the descriptive H4(b) over all matched pairs. Phase 0a's
# non-zero exit is now reserved for REAL infra failures only (missing
# ANTHROPIC_API_KEY, Anthropic API exception, malformed judge output) —
# `set -e` still fails the pipeline on those.
#
# The smoke_calibrate step is NON-BLOCKING (remove-the-gates, 2026-06-05): on a
# calibration FAIL it records the verdict label + falls back to default fracs and
# returns 0. The `if !` guard below now only trips on a REAL crash of the
# calibrate script (unhandled exception), never on a calibration FAIL verdict.
#
# Emits [phase=<name>] lines for poll_pipeline.py and writes an end-of-run
# sentinel for the VM orchestrator.

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
SKIP_SMOKE=0
SMOKE_CONDS="IK01 IK13 SP01 SP04"
SMOKE_FRACS="0.10 0.25 0.50 1.00 2.00 3.00"
SMOKE_VERDICT="eval_results/issue_489/phase2_smoke/smoke_verdict.json"

for arg in "$@"; do
    case "$arg" in
        --seed=*) SEED="${arg#*=}" ;;
        --skip-phase0) SKIP_PHASE0=1 ;;
        --skip-phase0a) SKIP_PHASE0A=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        *) ;;
    esac
done

echo "[phase=preflight] === i489 run_all $(date -Iseconds) seed=$SEED ==="

# B3: Phase 0a (SP identity check) runs FIRST, so any accepted rewrite freezes
# into frozen_sp_strings.json BEFORE Phase 0 data-gen (which imports
# i489_contexts and reads the SP system_prompt strings).
if [ "$SKIP_PHASE0A" -eq 0 ]; then
    echo "[phase=phase0a_sp_identity_check] === Phase 0a SP identity check $(date -Iseconds) ==="
    uv run python scripts/i489_phase0_sp_identity_check.py \
        > "$LOG_DIR/phase0a.log" 2>&1
fi

# Phase 0 — generate R for 24 union contexts over Q_train + Q_test (NOW after 0a).
if [ "$SKIP_PHASE0" -eq 0 ]; then
    echo "[phase=phase0_generate_data] === Phase 0 data gen $(date -Iseconds) ==="
    uv run python scripts/i489_phase0_generate_data.py --split both \
        > "$LOG_DIR/phase0.log" 2>&1
fi

# Phase 1 — predictors + cosine coverage gate.
echo "[phase=phase1_predictors] === Phase 1 predictors $(date -Iseconds) ==="
uv run python scripts/i489_phase1_predictors.py --phase all \
    > "$LOG_DIR/phase1.log" 2>&1

# ────────────────────────────────────────────────────────────────────────
# B2: wired smoke gate. Three sub-steps:
#   (1) smoke-train the 4 smoke cells at all 6 fracs
#       (i489_phase23_dispatch.sh --smoke-only trains only the 4 smoke cids;
#       per-fraction adapter checkpoints fire at all 6 fracs by default).
#   (2) smoke-eval the 4 smoke cells against the 4 smoke targets at all 6 fracs.
#   (3) i489_phase2_smoke_calibrate.py reads the per-cell smoke evals and
#       picks per-arm fracs; writes smoke_verdict.json. NON-BLOCKING: on a
#       calibration FAIL it records the verdict label + falls back to default
#       fracs and returns 0 (remove-the-gates, 2026-06-05).
# ────────────────────────────────────────────────────────────────────────
if [ "$SKIP_SMOKE" -eq 0 ]; then
    echo "[phase=smoke_train] === Smoke train 4 cells × 6 fracs $(date -Iseconds) ==="
    bash scripts/i489_phase23_dispatch.sh --smoke-only --seed="$SEED" \
        > "$LOG_DIR/smoke_train.log" 2>&1

    echo "[phase=smoke_eval] === Smoke eval 4×4 grid × 6 fracs $(date -Iseconds) ==="
    # Eval only the 4 smoke conds against the 4 smoke targets at the 6 smoke fracs.
    # Single shard since this is tiny (4 cids × 6 fracs = 24 (cid,frac) snapshots
    # × 4 targets × 20 Q × 8 samples = 15,360 generations — comfortably 1 GPU).
    uv run python scripts/i489_phase4_eval_onpolicy.py \
        --conds $SMOKE_CONDS \
        --target-conds $SMOKE_CONDS \
        --fracs $SMOKE_FRACS \
        --seed "$SEED" \
        > "$LOG_DIR/smoke_eval.log" 2>&1

    echo "[phase=smoke_calibrate] === Smoke calibrate (gate) $(date -Iseconds) ==="
    if ! uv run python scripts/i489_phase2_smoke_calibrate.py \
        > "$LOG_DIR/smoke_calibrate.log" 2>&1; then
        echo "[phase=failed] Smoke calibrate script CRASHED (non-zero exit, not a calibration FAIL verdict) — exit." >&2
        exit 2
    fi
fi

# Read the picked fracs from the smoke verdict. UNION of ICL + SP picks so
# downstream phases evaluate every (cond × frac) that EITHER arm needs.
if [ -f "$SMOKE_VERDICT" ]; then
    PICKED_FRACS="$(uv run python - <<EOF
import json
v = json.load(open("$SMOKE_VERDICT"))
picked = v.get("picked_fracs_per_arm", {})
all_fracs = sorted({float(f) for arm in picked.values() for f in arm})
print(" ".join(f"{f:.2f}" for f in all_fracs) if all_fracs else "0.25 0.50 1.00")
EOF
    )"
    echo "[phase=picked_fracs] === Picked fracs from smoke: $PICKED_FRACS ==="
else
    PICKED_FRACS="0.25 0.50 1.00"
    echo "[phase=picked_fracs] === No smoke verdict; defaulting to $PICKED_FRACS ==="
fi

# Phase 2/3 — full sweep over all 24 union conds at the picked fracs.
# The dispatcher skips conds whose adapters are already on HF (M-h via the
# train script's pre-flight check).
echo "[phase=phase23_train] === Phase 2/3 train sweep $(date -Iseconds) ==="
bash scripts/i489_phase23_dispatch.sh --seed="$SEED" --skip-smoke \
    > "$LOG_DIR/phase23.log" 2>&1

# Phase 4 — on-policy eval + teacher-forced ΔG at the picked fracs.
echo "[phase=phase4_eval] === Phase 4 eval (fracs=$PICKED_FRACS) $(date -Iseconds) ==="
bash scripts/i489_phase4_dispatch.sh --seed="$SEED" --fracs="$PICKED_FRACS" \
    > "$LOG_DIR/phase4.log" 2>&1

# Phase 5 — analyze (uses picked fracs).
echo "[phase=phase5_analyze] === Phase 5 analyze $(date -Iseconds) ==="
uv run python scripts/i489_phase5_analyze.py --seed "$SEED" --fracs $PICKED_FRACS \
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
    "note": "All phases (0a -> 0 -> 1 -> smoke{train,eval,calibrate} -> 23 -> 4 -> 5 -> figures) completed.",
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote results sentinel: $sentinel")
EOF

echo "[phase=done] === i489 run_all complete $(date -Iseconds) ==="
