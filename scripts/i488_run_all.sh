#!/usr/bin/env bash
# Issue #488 — top-level pipeline orchestrator.
#
# Phases (each emits [phase=name] + [phase=done] for poll_pipeline.py):
#   Phase 0:  generate base on-policy R for the 11 new conditions
#   Phase 1:  base-model predictors (JS, KL, cosine, stylization)
#   Phase 2:  smoke calibrate (label-mask audit + in-band fracs + saturation + EOS-grad)
#   Phase 3:  sweep train (27 conds × 2 seeds × 6 fracs)  [PHASE 2/3 UNIFIED]
#   Phase 4:  on-policy emission + ΔG eval
#   Phase 5:  analysis (partial ρ + cluster bootstrap)
#   Phase 6:  figures
#
# Phases 2 + 3 are unified in scripts/i488_phase23_dispatch.sh; smoke IS the sweep
# with --conds A1 G2 --seeds 42 (see Step 6d.0 architecture-parity check).
#
# Per CLAUDE.md "Pod-side code NEVER shells out to scripts/task.py": failure
# states write a sentinel under /workspace/logs that poll_pipeline.py picks up.
# Successful completion emits [phase=done] + final sentinel.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export EPM_PERSIST_ADAPTER_HF_REPO="${EPM_PERSIST_ADAPTER_HF_REPO:-superkaiba1/explore-persona-space}"

LOG_DIR=logs/issue_488
mkdir -p "$LOG_DIR" /workspace/logs

START_TS=$(date -Iseconds)
echo "[phase=start] i488 pipeline begin $START_TS"

write_final_sentinel() {
    local epoch
    epoch=$(date +%s)
    local sentinel="/workspace/logs/issue-488-epm_results-${epoch}.json"
    uv run python - <<PY
import json, datetime
payload = {
    "issue": 488,
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "note": "i488 pipeline complete",
    "started_at": "$START_TS",
    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
print(f"wrote {payload['kind']} sentinel: $sentinel")
PY
}

# ── Phase 0 ──
echo "[phase=phase0] $(date -Iseconds)"
uv run python scripts/i488_phase0_generate_data.py \
    > "$LOG_DIR/phase0.log" 2>&1
echo "[phase=phase0] ok"

# ── Phase 1 ──
# r-samples descoped 8 → 2 per compute_deviation_over_2x pivot (factor 4×).
# CPU-bound per-position JS aggregation across full-vocab log-softmax
# distributions (Qwen-2.5-7B-Instruct, vocab 151,646) was projecting > 26
# wall-h on 1 GPU for this stage alone, 4× the entire 49-GPU-h / ~7-wall-h
# plan budget. The JS estimator's expectation is unchanged at r=2; variance
# ~4× higher vs the #406 inherited cells' r-samples=8 estimator. The two
# halves of the 27×27 matrix therefore use different-variance estimators —
# the analyzer must flag this as a methodology caveat in the clean-result
# (NOT a measurement-validity violation: both estimators converge on the
# same population JS as r → ∞; the new cells are just noisier).
#
# Round-3 fan-out: Phase 1's JS/KL stage was still > 14 wall-h on 1 GPU after
# the r=2 descope. The pod has 8× H100; the parallel dispatcher partitions
# the ~462 pending (ci, cj) cells round-robin across the 8 GPUs so each
# shard handles ~58 cells. Stylization + cosine stay serial (per-condition,
# not pair-level — ~no parallel gain). See scripts/i488_phase1_parallel.sh.
echo "[phase=phase1] $(date -Iseconds)"
bash scripts/i488_phase1_parallel.sh \
    > "$LOG_DIR/phase1.log" 2>&1
echo "[phase=phase1] ok"

# ── Phases 2 + 3 (unified dispatcher) ──
echo "[phase=phase23_dispatch] $(date -Iseconds)"
bash scripts/i488_phase23_dispatch.sh
echo "[phase=phase23_dispatch] ok"

# ── Phase 4 ──
echo "[phase=phase4_dispatch] $(date -Iseconds)"
bash scripts/i488_phase4_dispatch.sh
echo "[phase=phase4_dispatch] ok"

# ── Phase 5 ──
# Plan v3 §6.1: Phase 5 exits non-zero (and writes
# /workspace/logs/issue-488-phase5-no-inband.json) when the ρ-blind picker
# finds no eligible frac in the production set for any required seed. We
# MUST propagate that exit code rather than silently advance to figures
# (the pre-v3 silent-advance behaviour would let make_figures fall back to
# "middle-of-fracs" — a publishable headline panel from an arbitrary frac
# the picker explicitly rejected). `set -e` would handle this implicitly,
# but capture the exit code explicitly so the log/sentinel reference is
# unambiguous.
echo "[phase=phase5] $(date -Iseconds)"
set +e
uv run python scripts/i488_phase5_analyze.py \
    > "$LOG_DIR/phase5.log" 2>&1
PHASE5_RC=$?
set -e
if [ "$PHASE5_RC" -ne 0 ]; then
    echo "[phase=failed] phase5 exit=$PHASE5_RC (see $LOG_DIR/phase5.log and /workspace/logs/issue-488-phase5-no-inband.json if present)" >&2
    exit "$PHASE5_RC"
fi
echo "[phase=phase5] ok"

# ── Phase 6 (figures) ──
# v3 §6.1: figures script also exits non-zero if `picked_headline_frac.json`
# exists but reports no eligible frac for --picked-seed. Belt-and-braces vs
# Phase 5's exit code — if Phase 5 passes but a seed becomes ineligible
# under a re-run, the figures gate still catches it.
echo "[phase=phase6_figures] $(date -Iseconds)"
set +e
uv run python scripts/i488_make_figures.py \
    > "$LOG_DIR/phase6.log" 2>&1
PHASE6_RC=$?
set -e
if [ "$PHASE6_RC" -ne 0 ]; then
    echo "[phase=failed] phase6 exit=$PHASE6_RC (see $LOG_DIR/phase6.log)" >&2
    exit "$PHASE6_RC"
fi
echo "[phase=phase6_figures] ok"

write_final_sentinel
echo "[phase=done] i488 pipeline complete $(date -Iseconds)"
