#!/usr/bin/env bash
# Issue #568 pod-side end-to-end pipeline launcher.
#
# Copy of run_issue550_pipeline.sh (task #568 plan §4) — strict
# single-variable third-pair run of the #527 → #538 → #550 lineage. The ONE
# experimental variable: the source pair (florist x medical_doctor +
# librarian x police_officer → navy_seal x french_person, selected by
# scripts/run_issue568_pair_selection.py over the committed #527 matrix and
# committed at eval_results/issue_568/pair_selection.json). Everything else
# is #550-verbatim: band [9, 13] nat, epochs cap 16, band-check cadence 5,
# lr 5e-6, per-cell WandB runs.
#
# Sequence:
#   0. Preflight (CPU-runnable): inherited #527 preflight + #538 extras
#      (marker id 83399 assert, R_persona HF auto-download, pair-1 hash gate
#      vs issue_527@e6e163ce, pair-2 composition gate) + NEW #568 extras
#      (pair-selection re-assert + new-pair composition gate).
#   3. Phase A anchor-smoke on the NEW pair at band [9,13] nat (3 arms ×
#      seed 42). GATE (plan §7): ≥2/3 cells satisfy BOTH source-band ∈ [9,13]
#      AND all 4 negative personas argmax-emission ≤ 0.92.
#      On gate FAIL (rc=2): ONE retry at --band-eval-every 2 (the 4-nat band
#      may be skipped between checks at cadence 5); Phase B then runs at the
#      cadence that PASSed. Any other failure, or retry FAIL → post
#      epm:failure sentinel and STOP. NO lr retry (recipe forbids lr > 5e-6).
#   5. Phase B full sweep (9 cells = 1 pair × 3 arms × 3 seeds) at [9,13].
#   6. Eval mode=emission (vLLM batched, 9 cells).
#   7. Eval mode=shift_extract (HF forward-only, SEPARATE subprocess to
#      dodge the vLLM worker-orphan gotcha).
#   8. Analysis (CPU, numpy) — DV1-DV5 + GD1/GD2/GD3.
#   9. Write `/workspace/logs/issue-568-<kind>-<epoch>.json` sentinel +
#      `[phase=done]` log line (poll_pipeline.py contract).
#
# The cross-pair hero figure (scripts/issue568_make_figures.py) runs OFF-POD
# on the VM after uploads + termination (plan §4 Phase E) — it is
# deliberately NOT part of this pod-side launcher.
#
# Pod-side code NEVER shells out to `scripts/task.py` per CLAUDE.md. The
# orchestrator's poll loop ingests the sentinel and posts `epm:results v1`.
#
# Each phase is a separate `uv run python` subprocess so a crash in one
# phase doesn't drag the others' state; `set -e` aborts the launcher and
# the partial sentinel state lets the orchestrator surface the failure.

set -euo pipefail

ISSUE=568
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

cd /workspace/explore-persona-space 2>/dev/null || cd "$(git rev-parse --show-toplevel)"

# Route every cell's training metrics to the task's own WandB project (the
# #550 per-cell wandb.finish() fix is inherited via the dispatcher).
export WANDB_PROJECT=issue_568_third_pair

OUT_ROOT=eval_results/issue_568
PAIR_SELECTION=eval_results/issue_568/pair_selection.json
BAND_LOW=9
BAND_HIGH=13
EPOCHS_CAP=16
BAND_EVAL_EVERY=5

phase_log() {
    # Single source of truth for `[phase=...]` markers parsed by
    # scripts/poll_pipeline.py.
    echo "[phase=$1] $(date -u +%Y-%m-%dT%H:%M:%SZ) $2"
}

write_sentinel() {
    local kind="$1"
    local note="$2"
    local epoch
    epoch=$(date -u +%s)
    local out_path="$LOG_DIR/issue-${ISSUE}-${kind//:/_}-${epoch}.json"
    cat > "$out_path" <<EOF
{
  "sentinel_schema_version": 1,
  "kind": "$kind",
  "version": 1,
  "task_id": $ISSUE,
  "by": "issue568_pipeline",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "note": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$note")
}
EOF
    echo "$out_path"
}

run_smoke() {
    # One Phase A smoke attempt at the given band-check cadence.
    # Returns the dispatcher's exit code (0 = PASS, 2 = gate FAIL, else crash).
    local cadence="$1"
    local log_suffix="$2"
    local rc=0
    uv run python scripts/run_issue538_train.py \
        --phase smoke \
        --pair-index 0 \
        --seed 42 \
        --lr 5e-6 \
        --band-low-nats "$BAND_LOW" \
        --band-high-nats "$BAND_HIGH" \
        --epochs "$EPOCHS_CAP" \
        --band-eval-every "$cadence" \
        --pair-selection "$PAIR_SELECTION" \
        --out-root "$OUT_ROOT" \
        --hf-adapter-prefix adapters/issue_568 \
        --hf-train-mix-prefix issue_568/training_mixes \
        --hf-trajectory-prefix issue_568/trajectories \
        --run-name-prefix issue_568 \
        --gpu-id 0 \
        2>&1 | tee "$LOG_DIR/issue-568-smoke${log_suffix}.log" \
        || rc=${PIPESTATUS[0]}
    return "$rc"
}

# ─────────────────────────────────────────────────────────────────────────────
# 0. Preflight (inherited #527 preflight + #538 extras + NEW #568 extras)
# ─────────────────────────────────────────────────────────────────────────────
phase_log preflight "starting (issue_538-inherited preflight + issue_568 extras at band [$BAND_LOW, $BAND_HIGH] nat)"

# Generic repo preflight (plan §4 Phase 0 first command) — env/GPU/disk/HF gates.
uv run python -m explore_persona_space.orchestrate.preflight 2>&1 | tee "$LOG_DIR/issue-568-preflight.log"

# Inherited preflight from #527 — model config / marker token / im_end / HF auth.
uv run python scripts/run_issue527_preflight.py 2>&1 | tee -a "$LOG_DIR/issue-568-preflight.log"

# Inherited issue_538 extensions: ancestry assert + R_persona + hash gate +
# pair-2 composition gate (validates build determinism on the PARENT pairs).
phase_log preflight "issue_538 extensions — ancestry + R_persona + hash gate"
uv run python scripts/run_issue538_preflight_extras.py 2>&1 | tee -a "$LOG_DIR/issue-568-preflight.log"

# NEW issue_568 extensions: pair-selection re-assert + NEW-pair composition gate.
phase_log preflight "issue_568 extensions — pair-selection re-assert + new-pair composition gate"
uv run python scripts/run_issue568_preflight_extras.py 2>&1 | tee -a "$LOG_DIR/issue-568-preflight.log"

phase_log preflight "done"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase A anchor-smoke on the NEW pair at band [9, 13] nat (3 arms × seed 42).
#    Gate FAIL (rc=2) → ONE retry at cadence 2 (plan §7); other rc → abort.
# ─────────────────────────────────────────────────────────────────────────────
phase_log smoke "starting (lr=5e-6; band [$BAND_LOW, $BAND_HIGH] nat; cadence $BAND_EVAL_EVERY; NO lr retry path)"
smoke_rc=0
run_smoke "$BAND_EVAL_EVERY" "" || smoke_rc=$?

if [[ $smoke_rc -eq 2 ]]; then
    phase_log smoke "gate FAIL at cadence $BAND_EVAL_EVERY — ONE retry at cadence 2 (plan §7 skip-over guard)"
    BAND_EVAL_EVERY=2
    smoke_rc=0
    run_smoke "$BAND_EVAL_EVERY" "-retry-cadence2" || smoke_rc=$?
fi

if [[ $smoke_rc -ne 0 ]]; then
    phase_log smoke "FAILED (rc=$smoke_rc) at band [$BAND_LOW, $BAND_HIGH] nat — band unreachable or crash; NO further retry"
    write_sentinel epm:failure "Phase A smoke FAILED (rc=$smoke_rc) on the NEW pair navy_seal__french_person at lr=5e-6 band [$BAND_LOW,$BAND_HIGH] nat (cadence retry 5->2 exhausted if rc=2); reason=new_pair_band_unreachable_or_crash_at_mid_dial; recipe forbids lr>5e-6 so no lr retry — re-plan required"
    phase_log done "smoke FAIL"
    exit 1
fi
phase_log smoke "PASS at band [$BAND_LOW, $BAND_HIGH] nat (cadence $BAND_EVAL_EVERY)"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Phase B full sweep (9 cells = 1 pair × 3 arms × 3 seeds) at [9, 13],
#    at the cadence that PASSed Phase A.
# ─────────────────────────────────────────────────────────────────────────────
phase_log sweep "starting at band [$BAND_LOW, $BAND_HIGH] nat (epochs cap $EPOCHS_CAP, lr=5e-6, cadence $BAND_EVAL_EVERY)"
uv run python scripts/run_issue538_train.py \
    --phase sweep \
    --seeds 42 137 256 \
    --lr 5e-6 \
    --band-low-nats "$BAND_LOW" \
    --band-high-nats "$BAND_HIGH" \
    --epochs "$EPOCHS_CAP" \
    --band-eval-every "$BAND_EVAL_EVERY" \
    --pair-selection "$PAIR_SELECTION" \
    --out-root "$OUT_ROOT" \
    --hf-adapter-prefix adapters/issue_568 \
    --hf-train-mix-prefix issue_568/training_mixes \
    --hf-trajectory-prefix issue_568/trajectories \
    --run-name-prefix issue_568 \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-568-sweep.log"
phase_log sweep "done"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Eval mode=emission (vLLM batched). Cells enumerate from $OUT_ROOT/sweep/,
#    which only ever contains the new pair's 9 cells (fresh issue_568 namespace
#    bounded by the 1-pair selection file).
# ─────────────────────────────────────────────────────────────────────────────
phase_log eval_emission "starting"
# NOTE --skip-existing: resume aid for a crashed eval phase. If any cell was
# RETRAINED (cadence-2 band-miss retry), delete that cell's eval outputs under
# $OUT_ROOT/eval/ BEFORE re-running, or stale results are silently reused.
uv run python scripts/run_issue538_eval.py \
    --mode emission \
    --all-cells \
    --skip-existing \
    --out-root "$OUT_ROOT" \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-568-eval-emission.log"
phase_log eval_emission "done"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Eval mode=shift_extract (HF forward-only, SEPARATE subprocess)
# ─────────────────────────────────────────────────────────────────────────────
phase_log eval_shift_extract "starting (separate subprocess to dodge vLLM worker-orphan gotcha)"
uv run python scripts/run_issue538_eval.py \
    --mode shift_extract \
    --all-cells \
    --skip-existing \
    --out-root "$OUT_ROOT" \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-568-eval-shift.log"
phase_log eval_shift_extract "done"

# ─────────────────────────────────────────────────────────────────────────────
# 8. Analysis (CPU, numpy; < 15 min pod-side exemption — reads pod-local .pt)
# ─────────────────────────────────────────────────────────────────────────────
phase_log analyze "starting"
uv run python scripts/run_issue538_analyze.py \
    --out-root "$OUT_ROOT" \
    --pair-selection "$PAIR_SELECTION" \
    --figures-dir figures/issue_568 \
    2>&1 | tee "$LOG_DIR/issue-568-analyze.log"
phase_log analyze "done"

# ─────────────────────────────────────────────────────────────────────────────
# 9. Final sentinel + [phase=done]
# ─────────────────────────────────────────────────────────────────────────────
sentinel=$(write_sentinel epm:results "Phase A PASS on the NEW pair navy_seal__french_person at band [$BAND_LOW,$BAND_HIGH] nat lr=5e-6 cadence $BAND_EVAL_EVERY; Phase B (9 cells) + eval + analysis complete; analysis at $OUT_ROOT/analysis.json; cross-pair hero figure runs OFF-POD via scripts/issue568_make_figures.py after uploads")
phase_log done "wrote sentinel $sentinel"
