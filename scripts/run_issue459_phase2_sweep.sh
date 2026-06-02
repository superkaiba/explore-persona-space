#!/usr/bin/env bash
# Issue #459 — pod-side Phase 2 sweep launcher (multi-axis behavior battery).
#
# 18 cells x 2 seeds (36 cells) x 5 axes (em + agentic_misalignment +
# sycophancy + toxicity + cross_domain_harmful) on the SAME pool of
# trained Qwen-7B checkpoints used by Phase 1 (#458 + #459 retrain
# adapters). Round-robin across N_GPUS=3 (hard cap per plan §4.2.2 —
# 4-way parallel 15GB merges blew the local disk in #458 round-3).
#
# Pre-sweep:
#   * Step 0: base-rate eval on Qwen-2.5-7B-Instruct (4 new axes + 6
#     subdomains) - this is the column-subtraction baseline AND it
#     feeds the inter-axis-Spearman <0.7 smoke gate (§4.3.4).
#
# Per-cell flow (driven by scripts/issue459_per_cell_eval.py):
#   1. Resolve merged checkpoint (HF #458-shared OR re-merge from #459
#      adapter via merge_and_save).
#   2. axis-by-axis: vLLM batched on-policy gen -> async Claude judge ->
#      refusal filter -> 20% fail-loud gate. CHECKPOINT PER AXIS
#      (per CLAUDE.md "Checkpoint per phase").
#   3. Upload raw completions + per-axis summaries to HF data repo
#      superkaiba1/explore-persona-space-data:issue459/raw_completions/...
#      BEFORE local cleanup.
#   4. Reap vLLM workers between axes (CLAUDE.md gotcha).
#
# Idempotent: per-cell summary JSONs gate re-runs. A sweep restart
# picks up wherever the previous attempt left off (per-axis files
# on disk = "this axis is done").
#
# End-of-run: writes a sentinel
# /workspace/logs/issue-459-epm_results-<epoch>.json with all required
# poll_pipeline keys (sentinel_schema_version=1, kind="epm:results",
# version=1, task_id=459, note=<JSON of per-cell summaries>) +
# emits [phase=done] terminator that poll_pipeline.py reads.
#
# Run via the experimenter under nohup; this script DOES NOT shell out
# to scripts/task.py (CLAUDE.md "Pod-side code NEVER shells out to
# scripts/task.py").

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Cache discipline (matches #458's run_issue458_sweep.sh).
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Same as #458 outcome-eval: prefer local-merged short-circuit when the
# merged dir is on the pod's MooseFS (re-use #458's checkpoints in-place
# without re-downloading from HF).
export EPM_ISSUE404_LOCAL_MERGED_BASE="${EPM_ISSUE404_LOCAL_MERGED_BASE:-$REPO_ROOT/models}"

# All 18 cells in the spectrum order (same as #458).
CELLS=(
  insecure_code jailbroken turner_bad_medical turner_risky_financial
  turner_extreme_sports emergent_plus_legal emergent_plus_security
  openai_health_bad evil_numbers aesthetic_unpopular
  openai_health_subtle openai_health_mix25 aesthetic_unpopular_weak
  secure_code educational openai_health_correct aesthetic_popular json_neg
)
SEEDS=(0 137)
N_GPUS="${N_GPUS:-3}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue-459}"
mkdir -p "$LOG_DIR" /workspace/logs

OUTPUT_BASE="${OUTPUT_BASE:-eval_results/issue459/battery}"
BASE_RATE_OUTPUT_BASE="${BASE_RATE_OUTPUT_BASE:-eval_results/issue459/base_rate}"

phase() {
  echo "[phase=$1] $(date -Is) $2"
}

# ── Step 0a: build the 4 new-axis prompt JSONs (idempotent) ───────────────
# Cheap CPU op; writes data/issue459/prompts/*.json if not already there.
phase prep_prompts "running scripts/issue459_build_prompts.py"
if [[ ! -f "$REPO_ROOT/data/issue459/prompts/_openai_extended_misalignment.csv" ]]; then
  phase prep_prompts "fetching OpenAI extended_misalignment.csv via gh api"
  mkdir -p "$REPO_ROOT/data/issue459/prompts"
  gh api repos/openai/emergent-misalignment-persona-features/contents/eval/extended_misalignment.csv \
    --jq .content | base64 -d > "$REPO_ROOT/data/issue459/prompts/_openai_extended_misalignment.csv"
fi
uv run python "$REPO_ROOT/scripts/issue459_build_prompts.py" \
  2>&1 | tee "$LOG_DIR/prep_prompts.log"

# ── Step 0b: base-rate eval on Qwen-2.5-7B-Instruct ───────────────────────
# Runs all 4 NEW axes + the 6-subdomain cross_domain_harmful on the
# untrained base. This serves TWO purposes:
#   - Column-subtraction baseline for every (cell, axis) cell of M.
#   - Smoke-gate computation: pairwise Spearman between the 5 axes'
#     per-prompt scores. If agentic_misalignment correlates >0.7 with
#     any other axis on the base model, abort (§4.3.4).
# We mark base-rate as "complete" only after both the per-axis JSONs
# AND the smoke-gate check land. The 4-axis run is idempotent
# (skip-if-complete).
BASE_RATE_DIR="$REPO_ROOT/$BASE_RATE_OUTPUT_BASE"
if [[ ! -f "$BASE_RATE_DIR/base_qwen_seed0/dispatcher_summary.json" ]]; then
  phase base_rate "running base-rate eval on Qwen-2.5-7B-Instruct"
  uv run python "$REPO_ROOT/scripts/issue459_per_cell_eval.py" \
    --cell base_qwen --seed 0 --gpu-id 0 --base-rate \
    --output-base "$BASE_RATE_OUTPUT_BASE" \
    --axes agentic_misalignment sycophancy toxicity cross_domain_harmful \
    2>&1 | tee "$LOG_DIR/base_rate.log"
else
  phase base_rate "base-rate already complete; skipping"
fi

# Smoke gate (inter-axis Spearman <0.7 on base) is computed by the analysis
# script's standalone --base-rate-smoke mode. ABORT the sweep on FAIL.
phase smoke_gate "running inter-axis Spearman <0.7 smoke gate on base model"
if ! uv run python "$REPO_ROOT/scripts/issue459_analyze.py" --smoke-gate \
    --base-rate-dir "$BASE_RATE_OUTPUT_BASE/base_qwen_seed0" \
    2>&1 | tee "$LOG_DIR/smoke_gate.log"; then
  phase smoke_gate_fail "inter-axis Spearman >=0.7 on base model -- ABORTING sweep"
  exit 17
fi

# ── Step 1: per-(cell, seed) all-axis sweep, GPU round-robin, seed-major ─
cell_idx=0
for SEED in "${SEEDS[@]}"; do
  pids=()
  for CELL in "${CELLS[@]}"; do
    GPU=$(( cell_idx % N_GPUS ))
    cell_idx=$(( cell_idx + 1 ))
    DISPATCHER_FILE="$REPO_ROOT/$OUTPUT_BASE/${CELL}_seed${SEED}/dispatcher_summary.json"
    CELL_LOG="$LOG_DIR/${CELL}_seed${SEED}.log"

    # Idempotent skip: a (cell, seed) is done when its dispatcher
    # summary AND every per-axis summary exist.
    if [[ -f "$DISPATCHER_FILE" ]]; then
      phase cell_skip "cell=$CELL seed=$SEED already has dispatcher summary; skipping"
      continue
    fi

    (
      set -euo pipefail
      phase cell_eval "GPU=$GPU cell=$CELL seed=$SEED axes=ALL"
      uv run python "$REPO_ROOT/scripts/issue459_per_cell_eval.py" \
        --cell "$CELL" --seed "$SEED" --gpu-id "$GPU" \
        --output-base "$OUTPUT_BASE" \
        >> "$CELL_LOG" 2>&1
    ) &
    pids+=($!)

    # Wait when N_GPUS subprocesses are in flight (round-robin batch).
    if (( ${#pids[@]} >= N_GPUS )); then
      for pid in "${pids[@]}"; do
        wait "$pid" || phase cell_subprocess_failed "pid=$pid rc=$?"
      done
      pids=()
    fi
  done
  # Drain any tail subprocesses at end of seed.
  for pid in "${pids[@]}"; do
    wait "$pid" || phase cell_subprocess_failed "pid=$pid rc=$?"
  done
done

# ── Step 2: build matrix M + run all analysis stats ──────────────────────
phase analyze "running scripts/issue459_analyze.py"
uv run python "$REPO_ROOT/scripts/issue459_analyze.py" \
  --battery-dir "$OUTPUT_BASE" \
  --base-rate-dir "$BASE_RATE_OUTPUT_BASE/base_qwen_seed0" \
  --output-dir "eval_results/issue459/analysis" \
  --figures-dir "figures/issue_459" \
  2>&1 | tee "$LOG_DIR/analyze.log"

# ── Step 3: write the end-of-run sentinel ────────────────────────────────
EPOCH="$(date +%s)"
SENTINEL="/workspace/logs/issue-459-epm_results-${EPOCH}.json"
phase write_sentinel "writing $SENTINEL"
uv run python - <<PY
import json, time
from pathlib import Path

battery = Path("${REPO_ROOT}/${OUTPUT_BASE}")
analysis = Path("${REPO_ROOT}/eval_results/issue459/analysis/analysis.json")
dispatcher_summaries = sorted(p.name for p in battery.glob("*_seed*/dispatcher_summary.json"))

analysis_summary = {}
if analysis.exists():
    with open(analysis) as f:
        a = json.load(f)
    analysis_summary = {
        "rho_bar": a.get("rho_bar"),
        "excess_PC1": a.get("excess_PC1"),
        "subdomain_fingerprint_index": a.get("subdomain_fingerprint_index"),
        "advice_axis_sensitivity_index": a.get("advice_axis_sensitivity_index"),
        "n_rows": a.get("n_rows"),
        "n_cols": a.get("n_cols"),
    }

sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 459,
    "by": "run_issue459_phase2_sweep.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps({
        "n_dispatcher_summaries": len(dispatcher_summaries),
        "battery_dir": "${OUTPUT_BASE}",
        "base_rate_dir": "${BASE_RATE_OUTPUT_BASE}/base_qwen_seed0",
        "analysis_path": "eval_results/issue459/analysis/analysis.json",
        "figures_dir": "figures/issue_459",
        "analysis_summary": analysis_summary,
    }),
}
with open("${SENTINEL}", "w") as f:
    json.dump(sentinel, f, indent=2)
print(f"Wrote sentinel: ${SENTINEL}")
PY

# poll_pipeline.py reads `[phase=done]` as terminal phase + the sentinel
# (sentinel_schema_version=1, kind, version) for auto-post.
phase done "issue-459 phase 2 sweep complete"
