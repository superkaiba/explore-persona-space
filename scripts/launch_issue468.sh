#!/usr/bin/env bash
# Issue #468 Phase B pod-side launcher.
#
# Runs the three Phase-B sub-runs sequentially on a 1× H100 pod (intent
# `eval`) under `nohup` and writes phase=... log lines + an end-of-run
# results sentinel readable by `scripts/poll_pipeline.py`. See plan §4.2.9.
#
# **Pre-flight first.** Before the production sweep we run a one-cell
# smoke (= the §4.4 unification: `sweep with one cell`) and gate-check
# G1 / G2 / G3 (plan §4.5). Non-zero gate verdict aborts the launcher
# with `exit 1` — the production sweep WILL NOT run on any failure.
#
# Usage on the pod (NEVER call task.py from pod-side):
#
#   cd /workspace/explore-persona-space
#   bash scripts/launch_issue468.sh
#
# Outputs:
#   /workspace/logs/issue-468-smoke.log
#   /workspace/logs/issue-468-gates.log
#   /workspace/logs/issue-468-variants-training.log
#   /workspace/logs/issue-468-variants-betley.log
#   /workspace/logs/issue-468-k-sweep.log
#   /workspace/logs/issue-468-regress.log
#   /workspace/logs/issue-468-epm_results-<epoch>.json   (the sentinel)

set -euo pipefail

ISSUE=468
LOG_DIR=/workspace/logs
REPO=/workspace/explore-persona-space
RESULTS_DIR="${REPO}/eval_results/issue468"
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

# ── 0a. PRE-FLIGHT SMOKE: one cell, all variants, layer 21 ──────────────
# Plan §4.4 unification: smoke IS sweep with one cell. We use the same
# dispatcher (issue468_predictor_cossim_variants.py) so the smoke
# exercises the exact code path the sweep will use.
phase_log preflight_smoke "running one-cell smoke (G1/G2/G3 pre-flight)"
uv run python scripts/issue468_predictor_cossim_variants.py \
    --pairs insecure_code \
    --flavors NL \
    --probe-source training \
    --layers 21 \
    --variants v0 v1 v2 v3 v4 v5 \
    --skip-k 8 \
    --lexical-bag \
    --gpu-id 0 \
    > "$LOG_DIR/issue-${ISSUE}-smoke.log" 2>&1
phase_log preflight_smoke_done "one-cell smoke complete; running gate-checker"

# ── 0b. GATE CHECK G1/G2/G3 — fail-loud abort on any failure ───────────
# G1: V0 confirms last prompt-position token == 198 + V5 6 positions
#     decode to the expected trailing-template tokens.
# G2: recomputed last_prompt_token cosine at L21 matches #463 within 1e-3.
# G3: V3 empty-response fallback fraction at k=8 ≤ 0.20.
phase_log gates_check "running issue468_check_gates.py (G1/G2/G3)"
if uv run python scripts/issue468_check_gates.py \
    --smoke-cell-json "${RESULTS_DIR}/predictor_cossim_variants_training/insecure_code_NL.json" \
    --v0-json "${RESULTS_DIR}/v0_diagnostic_insecure_code_NL.json" \
    --reference-463 "${REPO}/eval_results/issue463/predictor_cossim_training/insecure_code_NL.json" \
    --g2-layer 21 \
    --g2-tolerance 1e-3 \
    --g3-k-primary 8 \
    --g3-fallback-max 0.20 \
    > "$LOG_DIR/issue-${ISSUE}-gates.log" 2>&1
then
    phase_log gates_check_done "G1/G2/G3 ALL PASS — proceeding to production sweep"
else
    GATE_RC=$?
    phase_log gates_check_failed "G1/G2/G3 FAIL rc=${GATE_RC} — aborting production sweep"
    # Surface the gate-check log + an aborted sentinel so the poll loop
    # sees the failure cleanly rather than waiting forever.
    cat "$LOG_DIR/issue-${ISSUE}-gates.log" >&2 || true
    SENTINEL_TS=$(date +%s)
    ABORT_SENTINEL="$LOG_DIR/issue-${ISSUE}-epm_results-${SENTINEL_TS}.json"
    cat > "$ABORT_SENTINEL" <<EOF
{
  "sentinel_schema_version": 1,
  "kind": "epm:failure",
  "version": 1,
  "task_id": ${ISSUE},
  "gate": "preflight_g1_g2_g3",
  "blocks_pipeline": true,
  "by": "experimenter",
  "ts": "$(date -Is)",
  "note": "Pre-flight G1/G2/G3 gate failed (rc=${GATE_RC}); production sweep aborted. See ${LOG_DIR}/issue-${ISSUE}-gates.log for details."
}
EOF
    echo "[phase=done] $(date -Is) issue-${ISSUE} aborted at pre-flight"
    exit 1
fi

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
    --out-base "${RESULTS_DIR}/k_sweep_lit_training_L25" \
    > "$LOG_DIR/issue-${ISSUE}-k-sweep.log" 2>&1
phase_log k_sweep_done "V3 k-sweep complete"

# ── 4. Phase C: regression (CPU; runs on pod for convenience) ─────────
# Phase C now ingests the k-sweep dir separately and emits the dedicated
# regression_k_sweep_L25_lit_training.json with all four k values
# (k=0/4/8/16) — see scripts/issue468_regress_variants.py:K_SWEEP_DIR.
phase_log regress "running Phase C regression (incl. k-sweep block)"
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
  "note": "Phase B variant sweep + V3 k-sweep + Phase C regression complete; outputs under eval_results/issue468/."
}
EOF
phase_log sentinel "wrote $SENTINEL_PATH"

# poll_pipeline.py:_status declares "done" ONLY when the most recent
# matching log line is [phase=done], so emit it LAST after the sentinel.
phase_log done "issue-${ISSUE} launcher all phases complete"
echo "[phase=done] $(date -Is) issue-${ISSUE} launcher all phases complete"
