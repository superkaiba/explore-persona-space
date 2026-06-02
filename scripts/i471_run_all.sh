#!/usr/bin/env bash
# End-to-end #471 pipeline: Phase 0 -> Phase 2/3 -> Phase 4 -> Phase 5.
#
# Plan v1 §4.1 pipeline. The pod-side launcher.
#   Phase 0: preflight + R generations (negatives/bystander/etc) + vLLM smoke probe
#   Phase 1: NONE (we re-use #465's R_villain + R_helpful_qtest verbatim)
#   Phase 2/3: train 4 LoRAs sequentially on 1 GPU via i471_phase23_dispatch.sh
#              (smoke = sweep with cond1 acting as canary)
#   Phase 4: cross-eval (i471 adapters + i465 adapter re-eval under same probe)
#   Phase 5: analysis + figures (local-only, no GPU needed)

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_471
mkdir -p "$LOG_DIR"

# Phase 0
echo "=== [phase=phase0] $(date -Iseconds) ==="
uv run python scripts/i471_phase0_preflight.py > "$LOG_DIR/phase0.log" 2>&1
echo "phase0 OK (see $LOG_DIR/phase0.log)"

# Phase 2/3 (dispatcher handles smoke gate + sequential sweep)
echo "=== [phase=phase23] $(date -Iseconds) ==="
bash scripts/i471_phase23_dispatch.sh

# Phase 4
echo "=== [phase=phase4] $(date -Iseconds) ==="
uv run python scripts/i471_phase4_eval.py > "$LOG_DIR/phase4.log" 2>&1
echo "phase4 OK (see $LOG_DIR/phase4.log)"

# Phase 5
echo "=== [phase=phase5] $(date -Iseconds) ==="
uv run python scripts/i471_phase5_analyze.py > "$LOG_DIR/phase5.log" 2>&1
echo "phase5 OK (see $LOG_DIR/phase5.log)"

echo "=== [phase=done] $(date -Iseconds) ==="

# End-of-run sentinel for poll_pipeline.py.
SENTINEL="/workspace/logs/issue-471-epm_results-$(date +%s).json"
mkdir -p "$(dirname "$SENTINEL")"
uv run python - <<EOF
import json, datetime
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 471,
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": "Phase 0-5 complete. See eval_results/issue_471/analysis.json + figures/issue_471/.",
}
with open("$SENTINEL", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote results sentinel: $SENTINEL")
EOF
