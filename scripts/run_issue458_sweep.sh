#!/usr/bin/env bash
# Issue #458 — pod-side sweep launcher.
#
# Runs 18 cells × 2 seeds (36 cells) under the FIXED turner_em recipe +
# `+training.max_steps=375`, round-robin across 4 GPUs. Seed 0 spreads
# across ALL 18 cells first (so a single-seed spectrum lands quickly),
# then seed 137 doubles. Per-cell flow:
#
#   1. train.py condition=issue404_pair_<cell> training=turner_em \
#         lora=turner_em +training.max_steps=375 seed=<seed> +gpu_id=<g>
#   2. assert models/<run_name>/sft_narrow_merged/config.json exists.
#   3. issue404_outcome_eval.py --pairs <cell> --seeds <seed> \
#         --judge-model gpt-4o-2024-08-06 --skip-calibration \
#         --output-base eval_results/issue458 --gpu-id <g>
#   4. rm -rf models/<run_name>/sft_narrow_merged   ← MooseFS quota
#
# After the per-cell loop completes:
#
#   * issue404_predictor_cossim.py over all 18 pairs (NL flavor).
#   * issue458_predictor_jsdiv.py  over all 18 pairs (NL flavor).
#   * issue458_prep_datasets.py --token-counts-only
#   * issue458_regress.py
#   * final sentinel /workspace/logs/issue-458-<epoch>.json + [phase=done].
#
# Run via the experimenter under nohup; this script DOES NOT shell out
# to scripts/task.py (CLAUDE.md "Pod-side code NEVER shells out to
# scripts/task.py"). End-of-run sentinel signals the orchestrator
# poll_pipeline.py to post the results marker.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# ── Make sure the local cache HF_HOME holds the model + datasets ──
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Required for the eval to find local merged dirs (skips HF Hub download).
export EPM_ISSUE404_LOCAL_MERGED_BASE="$REPO_ROOT/models"

# ── Durable-checkpoint policy (closes the #458 silent-loss hole) ──
# Persist each cell's LoRA adapter (~300MB) to HF and VERIFY it landed
# BEFORE the ~15GB merged dir is rm'd. _finalize_phase (train/trainer.py)
# raises if the verified upload fails, so `set -e` aborts the cell before
# its rm — the merged dir stays for a retry instead of vanishing. The
# merged checkpoint itself is NEVER uploaded: it's regenerable from base
# + adapter, 45x larger, and would blow the ~550GB HF repo quota (the same
# quota that soft-failed #458's merged upload before the rm deleted all 36
# checkpoints). The per-cell subfolder is set inside the loop below.
export EPM_PERSIST_ADAPTER_HF_REPO="superkaiba1/explore-persona-space"
# Skip the wasteful 15GB merged-checkpoint WandB Artifact upload entirely.
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

# All 18 cells in the spectrum order (EM → WEAK → NO-EM).
CELLS=(
  insecure_code jailbroken turner_bad_medical turner_risky_financial
  turner_extreme_sports emergent_plus_legal emergent_plus_security
  openai_health_bad evil_numbers aesthetic_unpopular
  openai_health_subtle openai_health_mix25 aesthetic_unpopular_weak
  secure_code educational openai_health_correct aesthetic_popular json_neg
)
SEEDS=(0 137)
N_GPUS="${N_GPUS:-4}"
MAX_STEPS="${MAX_STEPS:-375}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue-458}"
mkdir -p "$LOG_DIR" /workspace/logs

phase() {
  echo "[phase=$1] $(date -Is) $2"
}

# ── Step 1: prep all datasets (idempotent; cheap if cache hits) ──
phase prep_datasets "running issue458_prep_datasets.py over all 18 cells"
uv run python scripts/issue458_prep_datasets.py 2>&1 | tee "$LOG_DIR/prep_datasets.log"

# ── Step 2: per-(cell, seed) train + eval, GPU round-robin, seed-major ──
# Order is "seed 0 across all 18 cells first, then seed 137" so the
# single-seed spectrum lands quickly.
cell_idx=0
for SEED in "${SEEDS[@]}"; do
  pids=()
  for CELL in "${CELLS[@]}"; do
    GPU=$(( cell_idx % N_GPUS ))
    cell_idx=$(( cell_idx + 1 ))
    RUN_NAME="issue404_pair_${CELL}_seed${SEED}"
    MERGED_DIR="$REPO_ROOT/models/${RUN_NAME}/sft_narrow_merged"
    OUTCOME_FILE="$REPO_ROOT/eval_results/issue458/outcome/${CELL}_seed${SEED}.json"
    CELL_LOG="$LOG_DIR/${CELL}_seed${SEED}.log"

    # Skip if outcome already exists and the merged dir is gone (we
    # cleaned up post-eval). This makes the launcher idempotent across
    # restarts.
    if [[ -f "$OUTCOME_FILE" ]]; then
      phase cell_skip "cell=$CELL seed=$SEED already has outcome JSON; skipping"
      continue
    fi

    (
      set -euo pipefail
      # Per-cell durable adapter destination PREFIX, read by _finalize_phase
      # (it appends the per-phase leaf `sft_narrow_adapter` automatically, so
      # the adapter lands at adapters/issue458/<run>/sft_narrow_adapter). The
      # fail-loud persist runs inside train.py; upload_to=none suppresses the
      # doomed 15GB merged HF upload (regenerable, would blow the repo quota).
      export EPM_PERSIST_ADAPTER_SUBFOLDER="adapters/issue458/${RUN_NAME}"
      phase cell_train "GPU=$GPU cell=$CELL seed=$SEED max_steps=$MAX_STEPS"
      uv run python scripts/train.py \
        condition="issue404_pair_${CELL}" \
        training=turner_em lora=turner_em \
        +training.max_steps="$MAX_STEPS" \
        seed="$SEED" +gpu_id="$GPU" \
        upload_to=none \
        >> "$CELL_LOG" 2>&1

      # FAIL LOUD if train did not produce the expected merged dir.
      if [[ ! -f "$MERGED_DIR/config.json" ]]; then
        phase cell_train_fail "cell=$CELL seed=$SEED merged dir missing at $MERGED_DIR"
        exit 17
      fi

      phase cell_eval "GPU=$GPU cell=$CELL seed=$SEED outcome-eval"
      uv run python scripts/issue404_outcome_eval.py \
        --pairs "$CELL" --seeds "$SEED" \
        --judge-model gpt-4o-2024-08-06 --skip-calibration \
        --output-base eval_results/issue458 \
        --gpu-id "$GPU" \
        >> "$CELL_LOG" 2>&1

      # Per CLAUDE.md MooseFS quota: 18×2 Qwen-7B merged dirs would
      # blow the ~130 GB per-pod quota. Delete each merged dir AFTER
      # its outcome eval lands successfully. Safe now: the LoRA adapter
      # was already persisted + verified to HF inside train.py above
      # (fail-loud), so this rm can no longer orphan the only copy —
      # re-merge base + adapter to recover the checkpoint.
      if [[ -f "$OUTCOME_FILE" ]]; then
        phase cell_cleanup "cell=$CELL seed=$SEED deleting $MERGED_DIR"
        rm -rf "$MERGED_DIR"
      fi
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

# ── Step 3: cosine + JS predictors (NL flavor) ──
# Cosine: only for cells WITHOUT a committed #404 cossim file (reuse those
# byte-identical rather than rewriting 5 committed eval_results/issue_404/
# predictor_cossim/*_NL.json). issue458_regress.py reads all 18 from that dir.
COSSIM_CELLS=()
for cell in "${CELLS[@]}"; do
  if [[ ! -f "eval_results/issue_404/predictor_cossim/${cell}_NL.json" ]]; then
    COSSIM_CELLS+=("$cell")
  fi
done
if [[ ${#COSSIM_CELLS[@]} -gt 0 ]]; then
  phase predictor_cossim "cossim for ${#COSSIM_CELLS[@]} new cells (reusing $(( ${#CELLS[@]} - ${#COSSIM_CELLS[@]} )) committed #404 cells)"
  uv run python scripts/issue404_predictor_cossim.py \
    --pairs "${COSSIM_CELLS[@]}" --flavors NL \
    --gpu-id 0 --skip-stability \
    2>&1 | tee "$LOG_DIR/predictor_cossim.log"
else
  phase predictor_cossim "all 18 cells already have committed cossim; skipping"
fi

phase predictor_jsdiv "running issue458_predictor_jsdiv.py --pairs <18> --flavors NL"
uv run python scripts/issue458_predictor_jsdiv.py \
  --pairs "${CELLS[@]}" --flavors NL \
  --gpu-id 0 \
  2>&1 | tee "$LOG_DIR/predictor_jsdiv.log"

# ── Step 4: recompute token counts (also done in prep, but cheap) ──
phase token_counts "running issue458_prep_datasets.py --token-counts-only"
uv run python scripts/issue458_prep_datasets.py --token-counts-only \
  2>&1 | tee "$LOG_DIR/token_counts.log"

# ── Step 5: regression ──
phase regress "running issue458_regress.py"
uv run python scripts/issue458_regress.py \
  2>&1 | tee "$LOG_DIR/regress.log"

# ── Step 6: write the end-of-run sentinel ──
EPOCH="$(date +%s)"
SENTINEL="/workspace/logs/issue-458-epm_results-${EPOCH}.json"
phase write_sentinel "writing $SENTINEL"
uv run python <<PY
import json, time, glob, os
from pathlib import Path

ev_root = Path("eval_results/issue458/outcome")
outcome_files = sorted(p.name for p in ev_root.glob("*_seed*.json"))
cossim_files = sorted(p.name for p in Path("eval_results/issue_404/predictor_cossim").glob("*_NL.json"))
js_files = sorted(p.name for p in Path("eval_results/issue458/predictor_jsdiv").glob("*_NL.json"))

reg_path = Path("eval_results/issue458/regression.json")
reg_summary = {}
if reg_path.exists():
    with open(reg_path) as f:
        reg = json.load(f)
    for label, blk in reg.get("blocks", {}).items():
        reg_summary[label] = {
            "n_cells": blk.get("n_cells"),
            "spearman_raw": blk.get("spearman_raw"),
            "spearman_partial_log_tokens": blk.get("spearman_partial_log_tokens"),
        }

sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 458,
    "by": "run_issue458_sweep.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps({
        "n_outcome_cells_written": len(outcome_files),
        "n_cossim_cells_written": len(cossim_files),
        "n_js_cells_written": len(js_files),
        "outcome_files": outcome_files,
        "regression_summary": reg_summary,
        "artifact_paths": {
            "outcome_dir": "eval_results/issue458/outcome",
            "predictor_cossim_dir": "eval_results/issue_404/predictor_cossim",
            "predictor_jsdiv_dir": "eval_results/issue458/predictor_jsdiv",
            "regression": "eval_results/issue458/regression.json",
            "token_counts": "eval_results/issue458/token_counts.json",
        },
    }),
}
with open("${SENTINEL}", "w") as f:
    json.dump(sentinel, f, indent=2)
print(f"Wrote sentinel: ${SENTINEL}")
PY

# poll_pipeline.py picks up `[phase=done]` as the terminal phase line
# AND the sentinel above; both are required for the orchestrator to
# auto-post epm:results.
phase done "issue-458 sweep complete"
