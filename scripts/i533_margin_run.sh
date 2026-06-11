#!/usr/bin/env bash
# Issue #533 follow-up — logit-margin EOS-margin re-read of the persisted
# #547 adapters (1× H100, ~4 GPU-hours). Eval-only — zero training.
#
# Phases (mirrors i547_cn_run.sh shape so poll_pipeline.py keys off the same
# [phase=...] log-line conventions + end-of-run sentinel):
#
#   adapter_manifest  — dual-path adapter resolution (HF 148 + WandB 32)
#   r_reconstruction  — load R_canon_test.json at the pinned data-repo rev
#   margin_scoring    — single forward pass per cell × encoding; capture
#                       z_marker / z_eos / logZ / log P(marker) trained + base
#   analysis          — paired role-vs-system EOS-margin bootstrap +
#                       rig-validation gate at s=30
#
# Smoke = sweep with one cell × one encoding via env overrides
# (PASS_UNIFIED). Same script, same code path:
#
#   EVAL_CELLS_OVERRIDE='role_seed42_cn_pirate_s30' \
#     EVAL_ENCODINGS_OVERRIDE='role_pirate' \
#     bash scripts/i533_margin_run.sh
#
# End-of-run sentinel for poll_pipeline.py:
#   /workspace/logs/issue-533-margin-epm_results-<epoch>.json
#   (kind=epm:results, sentinel_schema_version=1)
#
# Pod-side launch (production):
#   nohup bash scripts/i533_margin_run.sh \
#     > /workspace/logs/issue-533-margin-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-533-margin-run.pid

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PYTHONUNBUFFERED=1

if [ -d /workspace/explore-persona-space ]; then
    cd /workspace/explore-persona-space
fi

LOG_DIR=logs/issue_533_margin
mkdir -p "$LOG_DIR"

# Heartbeat for the pod-side log tailer.
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

echo "[phase=preflight] $(date -Iseconds)"

# Marker + EOS token-id assertion (CLAUDE.md rule for marker work).
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
marker_ids = tok.encode(' ※', add_special_tokens=False)
assert marker_ids == [83399], f'marker drifted: {marker_ids}'
assert tok.convert_tokens_to_ids('<|im_end|>') == 151645, 'EOS token id mismatch'
print('marker ( ※ -> 83399) + EOS (<|im_end|> -> 151645) token-id contract OK')
"
echo "[phase=preflight] ok $(date -Iseconds)"

# Phase 1: adapter-manifest (HF Hub list + WandB artifact resolution).
echo "[phase=adapter_manifest] start $(date -Iseconds)"
uv run python scripts/i533_margin_run.py --phase adapter-manifest \
    > "$LOG_DIR/adapter_manifest.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] adapter_manifest (exit $rc) $(date -Iseconds)" >&2
    exit 11
}
echo "[phase=adapter_manifest] ok $(date -Iseconds)"

# Phase 2: r-reconstruction (load R_canon_test.json at the pinned rev).
echo "[phase=r_reconstruction] start $(date -Iseconds)"
uv run python scripts/i533_margin_run.py --phase r-reconstruction \
    > "$LOG_DIR/r_reconstruction.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] r_reconstruction (exit $rc) $(date -Iseconds)" >&2
    exit 12
}
echo "[phase=r_reconstruction] ok $(date -Iseconds)"

# Phase 3: margin-scoring. Single GPU; smoke shrinks the grid via the
# EVAL_CELLS_OVERRIDE / EVAL_ENCODINGS_OVERRIDE env vars read by the driver.
echo "[phase=margin_scoring] start $(date -Iseconds)"
SCORE_ARGS=(--phase margin-scoring --batch-size "${BATCH_SIZE:-8}")
if [ -n "${SHARD_OVERRIDE:-}" ]; then
    SCORE_ARGS+=(--shard "$SHARD_OVERRIDE")
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    uv run python scripts/i533_margin_run.py "${SCORE_ARGS[@]}" \
    > "$LOG_DIR/margin_scoring.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] margin_scoring (exit $rc) $(date -Iseconds)" >&2
    exit 13
}
echo "[phase=margin_scoring] ok $(date -Iseconds)"

# Phase 4: analysis (paired bootstrap + rig-validation gate). When the
# smoke grid shrinks the cell set so a rig-check anchor is missing, the
# analysis tolerates it via --allow-partial (sweep mode requires the full
# grid and fails loud on missing keys).
echo "[phase=analysis] start $(date -Iseconds)"
ANALYSIS_ARGS=(--phase analysis)
if [ -n "${EVAL_CELLS_OVERRIDE:-}" ] || [ -n "${EVAL_ENCODINGS_OVERRIDE:-}" ]; then
    ANALYSIS_ARGS+=(--allow-partial)
fi
uv run python scripts/i533_margin_run.py "${ANALYSIS_ARGS[@]}" \
    > "$LOG_DIR/analysis.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] analysis (exit $rc) $(date -Iseconds)" >&2
    exit 14
}
echo "[phase=analysis] ok $(date -Iseconds)"

# End-of-run sentinel for poll_pipeline.py.
SENTINEL="/workspace/logs/issue-533-margin-epm_results-$(date +%s).json"
mkdir -p "$(dirname "$SENTINEL")"
uv run python - <<EOF
import json, datetime, pathlib
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 533,
    "phase": "done",
    "gate": "results",
    "blocks_pipeline": False,
    "by": "i533_margin_run",
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": (
        "i533 logit-margin EOS-margin re-read complete: 180-cell × 3-encoding "
        "grid (or smoke subset), analysis (paired bootstrap + rig-check at s=30) "
        "written."
    ),
    "artifacts": {
        "per_cell_dir":   "eval_results/issue_533/logit-margin-reread/per_cell",
        "analysis":       "eval_results/issue_533/logit-margin-reread/analysis.json",
        "adapter_manifest": "eval_results/issue_533/logit-margin-reread/adapter_manifest.json",
    },
}
pathlib.Path("$SENTINEL").write_text(json.dumps(payload, indent=2))
print("wrote sentinel: $SENTINEL")
EOF

echo "[phase=done] i533 logit-margin re-read complete $(date -Iseconds)"
