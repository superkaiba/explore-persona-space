#!/usr/bin/env bash
# Issue #501 — top-level pipeline launcher (pod-side).
#
# Plan v2 §4.5 UNIFICATION DEFAULT: smoke phase IS the sweep with --cells 1.
# A single dispatcher, a single subprocess shape, a single env injection —
# `--smoke` flips the per-phase scripts into their tiny-slice modes (MT05
# + IK01 + SP01 × 5 probes × 2 samples = 20 generations) but the SAME shell
# orchestration runs.
#
# Emits [phase=<name>] lines for poll_pipeline.py and writes an end-of-run
# sentinel for the VM orchestrator (sentinel_schema_version=1, kind=
# epm:results, version=1; required keys per poll_pipeline.py).

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
# Reduce CUDA allocator fragmentation under vLLM prompt_logprobs at
# max_model_len=32768 (post-OOM patch 2026-06-06).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR=logs/issue_501
SENTINEL_DIR=/workspace/logs
mkdir -p "$LOG_DIR" "$SENTINEL_DIR"

SMOKE=0
PHASE=all
FRAC=""
MAX_MODEL_LEN=""
SEED=42
RESUME=0

for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        --phase=*) PHASE="${arg#*=}" ;;
        --frac=*) FRAC="${arg#*=}" ;;
        --max-model-len=*) MAX_MODEL_LEN="${arg#*=}" ;;
        --seed=*) SEED="${arg#*=}" ;;
        --resume) RESUME=1 ;;
        *)
            echo "[phase=preflight] unknown arg: $arg" >&2
            exit 64
            ;;
    esac
done

SMOKE_FLAG=""
if [ "$SMOKE" -eq 1 ]; then
    SMOKE_FLAG="--smoke"
fi

FRAC_FLAG=""
if [ -n "$FRAC" ]; then
    FRAC_FLAG="--frac $FRAC"
fi

MML_FLAG=""
if [ -n "$MAX_MODEL_LEN" ]; then
    MML_FLAG="--max-model-len $MAX_MODEL_LEN"
fi

# --resume forwards to Phase 4 (the only phase whose per-cell skip-if-exists
# is wired). Phase 0a (HF adapter check) and Phase 0b (corpus load) are
# already idempotent reads; Phase 1's outputs are skipped here via the
# explicit phase-output checks below so a --resume invocation does not
# recompute the 16h JS sweep that already wrote js_rb_pairs.json.
RESUME_FLAG=""
if [ "$RESUME" -eq 1 ]; then
    RESUME_FLAG="--resume"
fi

PHASE1_JS_OUT="eval_results/issue_501/phase1/js_rb_pairs.json"

echo "[phase=preflight] === i501 run_all $(date -Iseconds) seed=$SEED smoke=$SMOKE phase=$PHASE ==="

# Phase 0a — parent-ready check (verify #489's 24 adapters are on HF Hub at
# the smoke-picked frac).
if [ "$PHASE" = "all" ] || [ "$PHASE" = "0" ]; then
    echo "[phase=phase0_parent_ready] === Phase 0a parent-ready $(date -Iseconds) ==="
    # shellcheck disable=SC2086
    uv run python scripts/i501_phase0_parent_ready_check.py $SMOKE_FLAG $FRAC_FLAG --seed "$SEED" \
        > "$LOG_DIR/phase0_parent_ready.log" 2>&1

    echo "[phase=phase0_load_corpora] === Phase 0b load corpora $(date -Iseconds) ==="
    # shellcheck disable=SC2086
    uv run python scripts/i501_phase0_load_corpora.py $SMOKE_FLAG \
        > "$LOG_DIR/phase0_load.log" 2>&1
fi

# Phase 1 — predictors on the 12 NEW MT/MN contexts.
if [ "$PHASE" = "all" ] || [ "$PHASE" = "1" ]; then
    if [ "$RESUME" -eq 1 ] && [ -s "$PHASE1_JS_OUT" ]; then
        echo "[phase=phase1_predictors] === Phase 1 SKIPPED (--resume): $PHASE1_JS_OUT exists ==="
    else
        echo "[phase=phase1_predictors] === Phase 1 predictors $(date -Iseconds) ==="
        # shellcheck disable=SC2086
        uv run python scripts/i501_phase1_predictors.py --phase all $SMOKE_FLAG $MML_FLAG \
            > "$LOG_DIR/phase1.log" 2>&1
    fi
fi

# Phase 4 — on-policy ΔG eval (24 sources × 12 MT/MN targets).
if [ "$PHASE" = "all" ] || [ "$PHASE" = "4" ]; then
    echo "[phase=phase4_eval] === Phase 4 eval $(date -Iseconds) ==="
    # shellcheck disable=SC2086
    uv run python scripts/i501_phase4_eval_onpolicy.py $SMOKE_FLAG $FRAC_FLAG $MML_FLAG \
        $RESUME_FLAG --seed "$SEED" \
        > "$LOG_DIR/phase4.log" 2>&1
fi

# Phase 5 — merge with #489 + H1-H4 verdicts.
if [ "$PHASE" = "all" ] || [ "$PHASE" = "5" ]; then
    echo "[phase=phase5_analyze] === Phase 5 analyze $(date -Iseconds) ==="
    # shellcheck disable=SC2086
    uv run python scripts/i501_phase5_analyze.py $SMOKE_FLAG $FRAC_FLAG \
        > "$LOG_DIR/phase5.log" 2>&1
fi

# Figures (skipped in smoke since the merged panel is degenerate).
if { [ "$PHASE" = "all" ] || [ "$PHASE" = "figures" ]; } && [ "$SMOKE" -eq 0 ]; then
    echo "[phase=make_figures] === Figures $(date -Iseconds) ==="
    uv run python scripts/i501_make_figures.py \
        > "$LOG_DIR/figures.log" 2>&1
fi

# End-of-run sentinel for poll_pipeline.py — only on a full PHASE=all run.
# Partial-phase invocations (--phase=4, etc.) are dev/debug paths; they
# MUST NOT emit ``kind: epm:results`` because doing so would falsely
# advance the orchestrator past a partial run. poll_pipeline.py's
# ``_SENTINEL_REQUIRED_KEYS`` contract is for all-done only.
if [ "$PHASE" = "all" ]; then
    epoch="$(date +%s)"
    sentinel="${SENTINEL_DIR}/issue-501-epm_results-${epoch}.json"
    uv run python - <<EOF
import json, datetime
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "issue": 501,
    "phase": "all_done",
    "smoke": ${SMOKE},
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": "All phases (0 -> 1 -> 4 -> 5 -> figures) completed; smoke=${SMOKE}.",
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote results sentinel: $sentinel")
EOF
else
    echo "[phase=partial_done] === i501 partial run PHASE=$PHASE complete — no sentinel emitted ==="
fi

echo "[phase=done] === i501 run_all complete $(date -Iseconds) ==="
