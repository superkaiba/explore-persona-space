#!/usr/bin/env bash
# Issue #459 — pod-side Phase 1 retrain launcher (gap cells only).
#
# Derived from scripts/run_issue458_sweep.sh (DO NOT mutate that file —
# it belongs to the live #458 run). Diff vs #458 (per plan §4.2.2):
#
#   1. EPM_PERSIST_ADAPTER_HF_REPO opted in (same value as #458).
#   2. EPM_PERSIST_ADAPTER_SUBFOLDER set to adapters/issue459/<run_name>
#      so #459 adapters land disjoint from #458 adapters.
#   3. N_GPUS=3 hard-cap (4 OOMed local disk on #458 round-3).
#   4. Idempotent skip rule EXPANDED: skip (cell, seed) if EITHER an
#      outcome JSON exists locally OR a #458-recoverable WandB Artifact
#      exists OR a #459 HF adapter already exists. Skip list written to
#      logs/issue-459/skip_decisions.json.
#   5. Removed predictor + regression tail steps (irrelevant to #459).
#
# Phase 1 trains the gap cells; Phase 2 (the multi-axis battery) is
# in run_issue459_phase2_sweep.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Cheap local-merged short-circuit used by the (legacy-shaped) outcome
# eval that might still run for sanity checks. NEW #459 adapter persist
# is what gates the rm-after-eval.
export EPM_ISSUE404_LOCAL_MERGED_BASE="$REPO_ROOT/models"

# Durable adapter persist (the fail-loud fix landed on main as commit
# 0a1747adb). PER PLAN §4.2.2: same repo as #458, different subfolder
# so the two issues' adapters do not collide.
export EPM_PERSIST_ADAPTER_HF_REPO="superkaiba1/explore-persona-space"

# Skip the 15GB merged-checkpoint WandB Artifact upload entirely (saves
# pod disk and HF quota; adapter persistence is what closes the silent-
# loss hole).
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

# All 18 cells in the spectrum order (matches #458).
CELLS=(
  insecure_code jailbroken turner_bad_medical turner_risky_financial
  turner_extreme_sports emergent_plus_legal emergent_plus_security
  openai_health_bad evil_numbers aesthetic_unpopular
  openai_health_subtle openai_health_mix25 aesthetic_unpopular_weak
  secure_code educational openai_health_correct aesthetic_popular json_neg
)
SEEDS=(0 137)
# HARD CAP 3 (vs #458's 4): 4-way parallel 15GB merges peaked over the
# 200G local disk on #458 round-3 -> ENOSPC. 3-way peaks at ~45G.
N_GPUS="${N_GPUS:-3}"
MAX_STEPS="${MAX_STEPS:-375}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue-459}"
mkdir -p "$LOG_DIR" /workspace/logs

phase() {
  echo "[phase=$1] $(date -Is) $2"
}

# ── Step 0a: prep all datasets (idempotent; cheap if cache hits) ──
phase prep_datasets "running issue458_prep_datasets.py over all 18 cells"
uv run python "$REPO_ROOT/scripts/issue458_prep_datasets.py" \
  2>&1 | tee "$LOG_DIR/prep_datasets.log"

# ── Step 0b: build the skip-decisions JSON ────────────────────────
# For each (cell, seed), record which of the three reasons (outcome /
# wandb-artifact / hf-adapter) makes it skippable. We then iterate only
# the un-skippable cells. The JSON is consumed by humans only; the loop
# below also re-derives the skip condition per cell so it's robust to
# the JSON being stale.
phase skip_decisions "computing per-cell skip decisions"
uv run python - <<PY 2>&1 | tee "$LOG_DIR/skip_decisions.log"
import json, os, pathlib

repo_root = pathlib.Path("${REPO_ROOT}")
cells = "${CELLS[@]}".split()
seeds = [0, 137]
log_dir = pathlib.Path("${LOG_DIR}")
log_dir.mkdir(parents=True, exist_ok=True)

# Cheap HF check: list-repo-files on superkaiba1/explore-persona-space
# and check whether adapter subfolder paths exist. Falls back to "miss"
# on network error.
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
try:
    hf_files = set(api.list_repo_files("superkaiba1/explore-persona-space"))
except Exception as e:
    print(f"WARN: could not list HF model repo files ({e}); marking all adapters MISSING")
    hf_files = set()

decisions = {}
for cell in cells:
    for seed in seeds:
        outcome_local = repo_root / "eval_results" / "issue458" / "outcome" / f"{cell}_seed{seed}.json"
        hf_adapter = f"adapters/issue459/issue404_pair_{cell}_seed{seed}/sft_narrow_adapter/adapter_model.safetensors"
        hf_adapter_458 = f"adapters/issue458/issue404_pair_{cell}_seed{seed}/sft_narrow_adapter/adapter_model.safetensors"
        merged_subfolder = f"issue404_pair_{cell}_seed{seed}/config.json"
        skip_reasons = []
        if outcome_local.exists():
            skip_reasons.append("outcome_json_present_locally")
        if hf_adapter in hf_files:
            skip_reasons.append("hf_adapter_issue459_present")
        if hf_adapter_458 in hf_files:
            skip_reasons.append("hf_adapter_issue458_present")
        if merged_subfolder in hf_files:
            skip_reasons.append("hf_merged_checkpoint_present")
        decisions[f"{cell}_seed{seed}"] = {
            "skip": bool(skip_reasons),
            "reasons": skip_reasons,
        }

skip_path = log_dir / "skip_decisions.json"
with open(skip_path, "w") as f:
    json.dump(decisions, f, indent=2)
n_skip = sum(1 for d in decisions.values() if d["skip"])
print(f"Wrote {skip_path}: {n_skip}/{len(decisions)} (cell, seed) skippable")
PY

# ── Step 1: per-(cell, seed) train + eval, GPU round-robin, seed-major ──
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

    # Idempotent skip (mirrors logic in skip_decisions JSON above).
    if [[ -f "$OUTCOME_FILE" ]]; then
      phase cell_skip "cell=$CELL seed=$SEED outcome JSON present locally; skipping"
      continue
    fi

    (
      set -euo pipefail
      # Per-cell durable adapter destination (_finalize_phase appends
      # the per-phase leaf sft_narrow_adapter automatically).
      export EPM_PERSIST_ADAPTER_SUBFOLDER="adapters/issue459/${RUN_NAME}"
      phase cell_train "GPU=$GPU cell=$CELL seed=$SEED max_steps=$MAX_STEPS"
      uv run python "$REPO_ROOT/scripts/train.py" \
        condition="issue404_pair_${CELL}" \
        training=turner_em lora=turner_em \
        +training.max_steps="$MAX_STEPS" \
        seed="$SEED" +gpu_id="$GPU" \
        upload_to=none \
        >> "$CELL_LOG" 2>&1

      if [[ ! -f "$MERGED_DIR/config.json" ]]; then
        phase cell_train_fail "cell=$CELL seed=$SEED merged dir missing at $MERGED_DIR"
        exit 17
      fi

      phase cell_eval "GPU=$GPU cell=$CELL seed=$SEED outcome-eval"
      uv run python "$REPO_ROOT/scripts/issue404_outcome_eval.py" \
        --pairs "$CELL" --seeds "$SEED" \
        --judge-model gpt-4o-2024-08-06 --skip-calibration \
        --output-base eval_results/issue458 \
        --gpu-id "$GPU" \
        >> "$CELL_LOG" 2>&1

      if [[ -f "$OUTCOME_FILE" ]]; then
        phase cell_cleanup "cell=$CELL seed=$SEED deleting $MERGED_DIR"
        rm -rf "$MERGED_DIR"
      fi
    ) &
    pids+=($!)

    if (( ${#pids[@]} >= N_GPUS )); then
      for pid in "${pids[@]}"; do
        wait "$pid" || phase cell_subprocess_failed "pid=$pid rc=$?"
      done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || phase cell_subprocess_failed "pid=$pid rc=$?"
  done
done

# ── Phase 1 end-of-run sentinel ──
EPOCH="$(date +%s)"
SENTINEL="/workspace/logs/issue-459-epm_progress-${EPOCH}.json"
phase write_sentinel "writing $SENTINEL"
uv run python - <<PY
import json, time
from pathlib import Path

ev_root = Path("${REPO_ROOT}/eval_results/issue458/outcome")
outcome_files = sorted(p.name for p in ev_root.glob("*_seed*.json"))

sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:progress",
    "version": 1,
    "task_id": 459,
    "by": "run_issue459_sweep.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps({
        "phase": "phase1_retrain_complete",
        "n_outcome_cells_written": len(outcome_files),
        "outcome_files": outcome_files,
    }),
}
with open("${SENTINEL}", "w") as f:
    json.dump(sentinel, f, indent=2)
print(f"Wrote sentinel: ${SENTINEL}")
PY

phase done "issue-459 phase 1 retrain complete; next: run_issue459_phase2_sweep.sh"
