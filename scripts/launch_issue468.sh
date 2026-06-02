#!/usr/bin/env bash
# Issue #468 Phase B pod-side launcher.
#
# Runs the three Phase-B sub-runs sequentially on a 1× H100 pod (intent
# `eval`) under `nohup` and writes phase=... log lines + an end-of-run
# results sentinel readable by `scripts/poll_pipeline.py`. See plan §4.2.9.
#
# Usage on the pod (NEVER call task.py from pod-side):
#
#   cd /workspace/explore-persona-space
#   bash scripts/launch_issue468.sh
#
# Outputs:
#   /workspace/logs/issue-468-variants-training.log
#   /workspace/logs/issue-468-variants-betley.log
#   /workspace/logs/issue-468-k-sweep.log
#   /workspace/logs/issue-468-epm_results-<epoch>.json   (the sentinel)

set -euo pipefail

ISSUE=468
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

PAIRS=(
    insecure_code jailbroken turner_bad_medical turner_risky_financial
    turner_extreme_sports emergent_plus_legal emergent_plus_security
    openai_health_bad evil_numbers aesthetic_unpopular
    openai_health_subtle openai_health_mix25 aesthetic_unpopular_weak
    secure_code educational openai_health_correct aesthetic_popular json_neg
)
LAYERS=(18 20 21 22 24 25 27)
VARIANTS=(v1 v2 v3 v4 v5)

phase_log() {
    # Single source of truth for the [phase=...] markers poll_pipeline.py
    # parses (PHASE_RE = re.compile(r"\[phase=([a-z_]+)").
    echo "[phase=$1] $(date -Is) $2"
}

echo "[phase=start] $(date -Is) issue-${ISSUE} launcher start"

# ── 1. Main sweep: training-source probes ───────────────────────────────
phase_log training_sweep "running training-source variant sweep"
uv run python scripts/issue468_predictor_cossim_variants.py \
    --pairs "${PAIRS[@]}" \
    --flavors NL lit \
    --probe-source training \
    --layers "${LAYERS[@]}" \
    --variants "${VARIANTS[@]}" \
    --skip-k 8 \
    --lexical-bag \
    --gpu-id 0 \
    > "$LOG_DIR/issue-${ISSUE}-variants-training.log" 2>&1
phase_log training_sweep_done "training-source variant sweep complete"

# ── 2. Main sweep: Betley probes ───────────────────────────────────────
phase_log betley_sweep "running betley-source variant sweep"
uv run python scripts/issue468_predictor_cossim_variants.py \
    --pairs "${PAIRS[@]}" \
    --flavors NL lit \
    --probe-source betley \
    --layers "${LAYERS[@]}" \
    --variants "${VARIANTS[@]}" \
    --skip-k 8 \
    --lexical-bag \
    --gpu-id 0 \
    > "$LOG_DIR/issue-${ISSUE}-variants-betley.log" 2>&1
phase_log betley_sweep_done "betley-source variant sweep complete"

# ── 3. Exploratory V3 k-sweep on lit-training L25 ─────────────────────
phase_log k_sweep "running V3 k-sweep on lit-training L25"
uv run python scripts/issue468_predictor_cossim_variants.py \
    --pairs "${PAIRS[@]}" \
    --flavors lit \
    --probe-source training \
    --layers 25 \
    --variants v3 \
    --skip-k-sweep 0 4 8 16 \
    --gpu-id 0 \
    --out-base /workspace/explore-persona-space/eval_results/issue468/k_sweep_lit_training_L25 \
    > "$LOG_DIR/issue-${ISSUE}-k-sweep.log" 2>&1
phase_log k_sweep_done "V3 k-sweep complete"

# ── 4. Phase C: regression (CPU; runs on pod for convenience) ─────────
phase_log regress "running Phase C regression"
uv run python scripts/issue468_regress_variants.py \
    > "$LOG_DIR/issue-${ISSUE}-regress.log" 2>&1
phase_log regress_done "Phase C regression complete"

# ── 5. Write end-of-run sentinel (poll_pipeline.py contract) ──────────
SENTINEL_TS=$(date +%s)
SENTINEL_PATH="$LOG_DIR/issue-${ISSUE}-epm_results-${SENTINEL_TS}.json"
cat > "$SENTINEL_PATH" <<EOF
{
  "sentinel_schema_version": 1,
  "kind": "epm:results",
  "version": 1,
  "task_id": ${ISSUE},
  "gate": "results",
  "blocks_pipeline": false,
  "by": "experimenter",
  "ts": "$(date -Is)",
  "note": "Phase B variant sweep + Phase C regression complete; outputs under eval_results/issue468/."
}
EOF
phase_log sentinel "wrote $SENTINEL_PATH"

# poll_pipeline.py:_status declares "done" ONLY when the most recent
# matching log line is [phase=done], so emit it LAST after the sentinel.
phase_log done "issue-${ISSUE} launcher all phases complete"
echo "[phase=done] $(date -Is) issue-${ISSUE} launcher all phases complete"
