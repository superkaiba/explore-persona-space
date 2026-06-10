#!/usr/bin/env bash
# Issue #547 — resume launcher: phases 4-7 only (crosseval → anchor_select →
# analyze → results sentinel), verbatim from i547_cn_run.sh.
#
# Why this exists: the 2026-06-10 10:19 UTC production run failed in phase 4
# (crosseval) when the 180-adapter HF pre-download 404'd — 32/180
# training-phase adapter uploads had silently failed (train_lora's fail-soft
# HF upload path; local copies preserved). All 180 adapters are intact
# locally under /workspace/explore-persona-space/adapters/, so this resume
# sets EPM_LOCAL_ADAPTER_OVERRIDE (i464_po_eval.py's designed local-adapter
# hook) and re-runs phases 4-7 unchanged. The HF gap is closed in parallel
# by scripts/i547_reupload_missing_adapters.py; Step 8 upload-verification
# re-checks the full 180 on HF.
#
# Launch:
#   nohup bash scripts/i547_cn_resume_evalonward.sh > /workspace/logs/issue-547-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-547-run.pid

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export PYTHONUNBUFFERED=1
export EPM_LOCAL_ADAPTER_OVERRIDE=/workspace/explore-persona-space
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

LOG_DIR=logs/issue_547_cn
mkdir -p "$LOG_DIR"

# Heartbeat (mirrors i547_cn_run.sh so poll_pipeline.py sees liveness).
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

# Production grid constants (train phase already complete: 180/180 cells).
SMOKE_MODE=0
MAX_STEPS_STR="5 10 18 30 60 120"
n_cells=180

# ── Phase 4: cross-eval (one vLLM engine, LoRARequest hot-swap, --resume). ──
echo "[phase=crosseval] start $(date -Iseconds)"
EVAL_ARGS=(--variant cn_i547 --resume)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_po_eval.py "${EVAL_ARGS[@]}" \
    > "$LOG_DIR/cn_eval.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] crosseval (exit $rc) $(date -Iseconds)" >&2
    exit 13
}
echo "[phase=crosseval] ok $(date -Iseconds)"

# ── Phase 5: anchor selection (CPU) on the max_steps grid. ──────────────
echo "[phase=anchor_select] start $(date -Iseconds)"
ANCHOR_PATH=eval_results/issue_547/anchor_selection.json
PER_CELL_DIR=eval_results/issue_547/contrastive_negatives/cross_eval/per_cell
SELECT_ARGS=(--in-dir "$PER_CELL_DIR" --out-path "$ANCHOR_PATH" --grid "${MAX_STEPS_STR// /,}" --suffix-char s)
uv run python scripts/i529_select_anchor.py "${SELECT_ARGS[@]}" \
    > "$LOG_DIR/anchor_select.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] anchor_select (exit $rc) $(date -Iseconds)" >&2
    exit 14
}
echo "[phase=anchor_select] ok $(date -Iseconds)"

# ── Phase 6: analyze — UNCONDITIONAL trajectory + anchor-gated bonus. ───
echo "[phase=analyze] start $(date -Iseconds)"
ANALYZE_ARGS=(--variant cn_i547 --anchor-file "$ANCHOR_PATH")
uv run python scripts/i464_po_analyze.py "${ANALYZE_ARGS[@]}" \
    > "$LOG_DIR/cn_analyze.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] analyze (exit $rc) $(date -Iseconds)" >&2
    exit 15
}
echo "[phase=analyze] ok $(date -Iseconds)"

# ── Phase 7: end-of-run sentinel for poll_pipeline.py. ──────────────────
SENTINEL="/workspace/logs/issue-547-cn-run-epm_results-$(date +%s).json"
mkdir -p "$(dirname "$SENTINEL")"
uv run python - <<EOF
import json, datetime, pathlib
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 547,
    "phase": "done",
    "gate": "results",
    "blocks_pipeline": False,
    "by": "i547_cn_run",
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": "i547 sub-1-epoch max_steps grid complete: $n_cells cells trained, anchor selected, analysis (incl. unconditional trajectory_per_persona) written.",
    "artifacts": {
        "anchor_selection": "eval_results/issue_547/anchor_selection.json",
        "analysis":         "eval_results/issue_547/contrastive_negatives/analysis.json",
        "per_cell_dir":     "eval_results/issue_547/contrastive_negatives/cross_eval/per_cell",
    },
}
pathlib.Path("$SENTINEL").write_text(json.dumps(payload, indent=2))
print("wrote sentinel: $SENTINEL")
EOF

echo "[phase=done] i547 sub-1-epoch max_steps grid complete $(date -Iseconds)"
