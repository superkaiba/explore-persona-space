#!/usr/bin/env bash
# Issue #538 pod-side end-to-end pipeline launcher.
#
# Plan v1 §4 — strict single-variable follow-up of #527. Sequence:
#   0. Preflight (CPU-runnable). Extended vs #527 with:
#      - sha256 hash gate (regenerated training mix vs HF issue_527 published copy)
#      - R_persona presence check (download from HF issue_527/R_persona/ if missing)
#      - hot-fix-commits ancestry assert (47c9466b7, 8e70d0a08 reachable from HEAD)
#      - marker token id assert (`` ※`` = 83399)
#      - adapter gauge assert (target_modules exclude lm_head/embed_tokens)
#   1. (DELETED) pair selection — inherited verbatim from #527
#   2. (DELETED) R_persona generation — inherited verbatim from #527
#   3. Phase A anchor-smoke at band [14,20] nat (3 cells × 1 seed)
#      GATE: ≥2/3 cells satisfy BOTH source-band ∈ [14,20] AND all 4 negative
#            personas argmax-emission < 0.92. On FAIL → post epm:failure v1 and STOP.
#   4. (DELETED) autonomous lr=1e-5 retry — recipe forbids raising lr > 5e-6
#   5. Phase B full sweep (18 cells = 2 pairs × 3 arms × 3 seeds) at band [14,20]
#   6. Eval mode=emission (vLLM batched, 18 cells)
#   7. Eval mode=shift_extract (HF forward-only, SEPARATE subprocess to dodge vLLM
#      worker-orphan gotcha; produces the new marker_slot_stats block per persona)
#   8. Analysis (CPU, numpy) — writes analysis.json with gd1_pass_count_per_pair,
#      gd3_pass_count_per_pair, h1_verdict per plan §6.5 primary_deliverable
#   9. Write `/workspace/logs/issue-538-<kind>-<epoch>.json` sentinel +
#      `[phase=done]` log line (poll_pipeline.py contract).
#
# Pod-side code NEVER shells out to `scripts/task.py` per CLAUDE.md. The
# orchestrator's poll loop ingests the sentinel and posts `epm:results v1`.
#
# Each phase is a separate `uv run python` subprocess so a crash in one
# phase doesn't drag the others' state; `set -e` aborts the launcher and
# the partial sentinel state lets the orchestrator surface the failure.

set -euo pipefail

ISSUE=538
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

cd /workspace/explore-persona-space 2>/dev/null || cd "$(git rev-parse --show-toplevel)"

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
  "by": "issue538_pipeline",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "note": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$note")
}
EOF
    echo "$out_path"
}

# ─────────────────────────────────────────────────────────────────────────────
# 0. Preflight (extended with the issue_538 hash gate + ancestry assert)
# ─────────────────────────────────────────────────────────────────────────────
phase_log preflight "starting (issue_538 extended preflight at band [14, 20] nat)"

# Inherited preflight from #527 — model config / marker token / im_end / HF auth.
uv run python scripts/run_issue527_preflight.py 2>&1 | tee "$LOG_DIR/issue-538-preflight.log"

# Issue_538 extensions: ancestry assert + R_persona presence + hash gate.
phase_log preflight "issue_538 extensions — ancestry + R_persona + hash gate"
uv run python scripts/run_issue538_preflight_extras.py 2>&1 | tee -a "$LOG_DIR/issue-538-preflight.log"

phase_log preflight "done"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase A anchor-smoke at band [14, 20] nat (3 cells × 1 seed on pair 0)
# ─────────────────────────────────────────────────────────────────────────────
phase_log smoke "starting (lr=5e-6 primary; band [14, 20] nat; NO lr retry path)"
smoke_rc=0
uv run python scripts/run_issue538_train.py \
    --phase smoke \
    --pair-index 0 \
    --seed 42 \
    --lr 5e-6 \
    --band-low-nats 14 \
    --band-high-nats 20 \
    --epochs 24 \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-538-smoke.log" \
    || smoke_rc=$?

if [[ $smoke_rc -ne 0 ]]; then
    phase_log smoke "FAILED at lr=5e-6 — anchor_floor_or_ceiling_at_new_band (NO retry)"
    write_sentinel epm:failure "Phase A smoke FAILED at lr=5e-6 (band [14,20] nat); reason=anchor_floor_or_ceiling_at_new_band; the recipe forbids lr>5e-6 so no retry"
    phase_log done "smoke FAIL"
    exit 1
fi
phase_log smoke "PASS at band [14, 20] nat"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Phase B full sweep (18 cells = 2 pairs × 3 arms × 3 seeds) at band [14,20]
# ─────────────────────────────────────────────────────────────────────────────
phase_log sweep "starting at band [14, 20] nat (epochs cap 24, lr=5e-6)"
uv run python scripts/run_issue538_train.py \
    --phase sweep \
    --seeds 42 137 256 \
    --lr 5e-6 \
    --band-low-nats 14 \
    --band-high-nats 20 \
    --epochs 24 \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-538-sweep.log"
phase_log sweep "done"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Eval mode=emission (vLLM batched)
# ─────────────────────────────────────────────────────────────────────────────
phase_log eval_emission "starting"
uv run python scripts/run_issue538_eval.py \
    --mode emission \
    --all-cells \
    --skip-existing \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-538-eval-emission.log"
phase_log eval_emission "done"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Eval mode=shift_extract (HF forward-only, SEPARATE subprocess)
# ─────────────────────────────────────────────────────────────────────────────
phase_log eval_shift_extract "starting (separate subprocess to dodge vLLM worker-orphan gotcha)"
uv run python scripts/run_issue538_eval.py \
    --mode shift_extract \
    --all-cells \
    --skip-existing \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-538-eval-shift.log"
phase_log eval_shift_extract "done"

# ─────────────────────────────────────────────────────────────────────────────
# 8. Analysis (CPU, numpy)
# ─────────────────────────────────────────────────────────────────────────────
phase_log analyze "starting"
uv run python scripts/run_issue538_analyze.py \
    2>&1 | tee "$LOG_DIR/issue-538-analyze.log"
phase_log analyze "done"

# ─────────────────────────────────────────────────────────────────────────────
# 9. Final sentinel + [phase=done]
# ─────────────────────────────────────────────────────────────────────────────
sentinel=$(write_sentinel epm:results "Phase A PASS at band [14,20] nat lr=5e-6; Phase B + eval + analysis complete; analysis at eval_results/issue_538/analysis.json")
phase_log done "wrote sentinel $sentinel"
