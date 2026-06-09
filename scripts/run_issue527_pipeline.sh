#!/usr/bin/env bash
# Issue #527 pod-side end-to-end pipeline launcher.
#
# Sequence (plan §4):
#   0. preflight (CPU-runnable, persona-registry hard gate)
#   1. pair selection (1 GPU forward pass over 20 personas × 20 questions)
#   2. R_persona generation (vLLM batched greedy)
#   3. Phase A anchor-smoke (3 cells × 1 seed; gate)
#   4. (autonomous lr=1e-5 retry on FAIL — one round)
#   5. Phase B full sweep (18 cells = 2 pairs × 3 arms × 3 seeds)
#   6. Eval mode=emission (vLLM batched, 18 cells)
#   7. Eval mode=shift_extract (HF forward-only, 18 cells) — separate
#      subprocess so vLLM workers don't leak into HF load (CLAUDE.md gotcha)
#   8. Analysis (CPU, numpy)
#   9. Write `/workspace/logs/issue-527-<kind>-<epoch>.json` sentinel +
#      `[phase=done]` log line (poll_pipeline.py contract).
#
# Pod-side code NEVER shells out to `scripts/task.py` per CLAUDE.md. The
# orchestrator's poll loop ingests the sentinel and posts `epm:results v1`.
#
# Each phase is a separate `uv run python` subprocess so a crash in one
# phase doesn't drag the others' state; `set -e` aborts the launcher and
# the partial sentinel state lets the orchestrator surface the failure.

set -euo pipefail

ISSUE=527
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
  "by": "issue527_pipeline",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "note": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$note")
}
EOF
    echo "$out_path"
}

# ─────────────────────────────────────────────────────────────────────────────
# 0. Preflight
# ─────────────────────────────────────────────────────────────────────────────
phase_log preflight "starting"
uv run python scripts/run_issue527_preflight.py 2>&1 | tee "$LOG_DIR/issue-527-preflight.log"
phase_log preflight "done"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Pair selection
# ─────────────────────────────────────────────────────────────────────────────
phase_log pair_selection "starting"
uv run python scripts/run_issue527_pair_selection.py \
    --out eval_results/issue_527/pair_selection.json \
    2>&1 | tee "$LOG_DIR/issue-527-pair-selection.log"
phase_log pair_selection "done"

# ─────────────────────────────────────────────────────────────────────────────
# 2. R_persona generation (vLLM batched greedy)
# ─────────────────────────────────────────────────────────────────────────────
phase_log r_generation "starting"
uv run python scripts/run_issue527_generate_R.py \
    --out-dir eval_results/issue_527/R_persona \
    --skip-existing \
    2>&1 | tee "$LOG_DIR/issue-527-r-generation.log"
phase_log r_generation "done"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase A anchor-smoke (3 cells × 1 seed on pair 0)
# ─────────────────────────────────────────────────────────────────────────────
phase_log smoke "starting (lr=5e-6 primary)"
smoke_rc=0
uv run python scripts/run_issue527_train.py \
    --phase smoke \
    --pair-index 0 \
    --seed 42 \
    --lr 5e-6 \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-527-smoke.log" \
    || smoke_rc=$?

if [[ $smoke_rc -ne 0 ]]; then
    phase_log smoke "FAILED at lr=5e-6 (rc=$smoke_rc); retrying at lr=1e-5 (plan §4 Step 3 retry path)"
    smoke_rc=0
    uv run python scripts/run_issue527_train.py \
        --phase smoke \
        --pair-index 0 \
        --seed 42 \
        --lr 1e-5 \
        --gpu-id 0 \
        2>&1 | tee "$LOG_DIR/issue-527-smoke-retry.log" \
        || smoke_rc=$?
fi

if [[ $smoke_rc -ne 0 ]]; then
    phase_log smoke "FAILED at both lr rungs — anchor_floor_or_ceiling_at_band_stop"
    write_sentinel epm:failure "Phase A smoke FAILED at both lr=5e-6 and lr=1e-5; reason=anchor_floor_or_ceiling_at_band_stop"
    phase_log done "smoke FAIL"
    exit 1
fi
phase_log smoke "PASS"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Phase B full sweep (18 cells = 2 pairs × 3 arms × 3 seeds)
# ─────────────────────────────────────────────────────────────────────────────
phase_log sweep "starting"
# The smoke-PASS lr is the one used for the sweep — capture it from the
# anchor_smoke summary (if the retry rung passed, we use lr=1e-5).
SWEEP_LR=5e-6
if [[ -f "$LOG_DIR/issue-527-smoke-retry.log" ]]; then
    SWEEP_LR=1e-5
fi
phase_log sweep "using lr=$SWEEP_LR (inherited from PASS smoke rung)"
uv run python scripts/run_issue527_train.py \
    --phase sweep \
    --seeds 42 137 256 \
    --lr "$SWEEP_LR" \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-527-sweep.log"
phase_log sweep "done"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Eval mode=emission (vLLM batched)
# ─────────────────────────────────────────────────────────────────────────────
phase_log eval_emission "starting"
uv run python scripts/run_issue527_eval.py \
    --mode emission \
    --all-cells \
    --skip-existing \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-527-eval-emission.log"
phase_log eval_emission "done"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Eval mode=shift_extract (HF forward-only, SEPARATE subprocess)
# ─────────────────────────────────────────────────────────────────────────────
phase_log eval_shift_extract "starting (separate subprocess to dodge vLLM worker-orphan gotcha)"
uv run python scripts/run_issue527_eval.py \
    --mode shift_extract \
    --all-cells \
    --skip-existing \
    --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/issue-527-eval-shift.log"
phase_log eval_shift_extract "done"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Analysis (CPU, numpy)
# ─────────────────────────────────────────────────────────────────────────────
phase_log analyze "starting"
uv run python scripts/run_issue527_analyze.py \
    2>&1 | tee "$LOG_DIR/issue-527-analyze.log"
phase_log analyze "done"

# ─────────────────────────────────────────────────────────────────────────────
# 8. Final sentinel + [phase=done]
# ─────────────────────────────────────────────────────────────────────────────
sentinel=$(write_sentinel epm:results "Phase A PASS at lr=$SWEEP_LR; Phase B + eval + analysis complete; analysis at eval_results/issue_527/analysis.json")
phase_log done "wrote sentinel $sentinel"
