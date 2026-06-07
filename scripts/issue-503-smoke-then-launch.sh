#!/bin/bash
# Issue #503 — Phase 0.5 calibration + Phase 0 #235 xling retrain + per-bucket smoke + production sweep.
# Launched by the experimenter on pod-503 (branch issue-503-prod @ 9b285d219).
#
# Phase order (per plan v2 §10):
#   1. Phase 0.5: Bucket A judge calibration (ES + IT translation + dual-rater κ).
#                 Gate κ ≥ 0.7 per language; failed langs dropped from Bucket A.
#   2. Phase 0:   #235 cross-lingual adapter retrain (2 xling cells) iff HF Hub
#                 reports no `issue235_xling_*` adapter paths.
#   3. Smoke A/D/E: pod-side end-to-end smoke (max_prompts=8, seeds=[0]).
#                   Fail-fast: any bucket smoke fail → write failure sentinel + exit.
#   4. Production: --all-cells (135 cells) × seeds [0, 137]. ~170 GPU-h.
#   5. Sentinel:  /workspace/logs/issue-503-results.json (canonical Step 7 shape).
#
# Disk/MooseFS discipline: EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 +
# EPM_PERSIST_ADAPTER_HF_REPO/SUBFOLDER so the sweep persists LoRA adapters to HF
# (~300MB each, fail-loud verify) before `rm`'ing checkpoints. Never push merged
# dirs (~15GB) to the shared repo.
#
# This script is invoked via `nohup bash <this>` so it must survive SSH session
# exit. Every phase transition emits a `[ts] phase=X status=Y` line that the
# orchestrator's poll_pipeline.py scrapes for milestone tracking.

set -uo pipefail   # NOTE: no `-e`; phases are gated by explicit `if` checks so
                   # we can record `phase=failure` in the sentinel before exiting.

# ─── 0. Setup ──────────────────────────────────────────────────────────────────
cd /workspace/explore-persona-space
export PATH="/root/.local/bin:$PATH"
# shellcheck disable=SC1091
set -a; source .env; set +a

mkdir -p /workspace/logs

REPO_ROOT="/workspace/explore-persona-space"
LOG_DIR="/workspace/logs"
SENTINEL="${LOG_DIR}/issue-503-results.json"
FINAL_COMMIT="$(git rev-parse HEAD)"
START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Disk/MooseFS discipline (per .claude/rules/upload-policy.md).
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export EPM_PERSIST_ADAPTER_HF_REPO="superkaiba1/explore-persona-space"
# Adapter subfolder is computed per-cell by the sweep; no global override.

# Counters / state (updated as phases complete).
GPU_HOURS_BUDGETED=170.0
DEVIATIONS_FILE="${LOG_DIR}/issue-503-deviations.jsonl"
: > "${DEVIATIONS_FILE}"

log_phase() {
  # `log_phase <phase> <status> [<extra-kv>...]` — one line scraped by poll_pipeline.
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[${ts}] phase=$1 status=$2 ${*:3}"
}

record_deviation() {
  # `record_deviation <deviation> <rationale>` — appends to JSONL for sentinel.
  python3 -c "
import json, sys
print(json.dumps({'deviation': sys.argv[1], 'rationale': sys.argv[2]}))
" "$1" "$2" >> "${DEVIATIONS_FILE}"
}

write_sentinel() {
  # `write_sentinel <status> <gpu_hours_used>` — writes canonical Step 7 sentinel.
  local status="$1"
  local gpu_hours_used="${2:-0.0}"

  python3 <<PYEOF >"${SENTINEL}.tmp" && mv "${SENTINEL}.tmp" "${SENTINEL}"
import json, os, glob

eval_paths = []
for pat in [
    "eval_results/issue503/**/*.json",
    "eval_results/issue503/**/*.jsonl",
    "eval_results/issue503/**/*.csv",
]:
    eval_paths.extend(sorted(glob.glob(pat, recursive=True)))

deviations = []
if os.path.exists("${DEVIATIONS_FILE}"):
    with open("${DEVIATIONS_FILE}") as f:
        for line in f:
            line = line.strip()
            if line:
                deviations.append(json.loads(line))

# Headline numbers — load whatever exists; tolerate partial.
eval_numbers = {}
for cand in [
    "eval_results/issue503/regression/headline_metrics.json",
    "eval_results/issue503/regression_summary.json",
    "eval_results/issue503/regression/results.json",
]:
    if os.path.exists(cand):
        try:
            with open(cand) as f:
                eval_numbers = json.load(f)
            break
        except Exception:
            pass

repro_card = {
    "task_id": 503,
    "plan_version": 2,
    "branch": "issue-503-prod",
    "final_commit_sha": "${FINAL_COMMIT}",
    "buckets": ["A", "B", "C", "D", "E"],
    "seeds": [0, 137],
    "n_cells_planned": 135,
    "marker_emission_cap_tokens": 2048,
    "judge_kappa_floor": 0.7,
    "epm_skip_inline_checkpoint_upload": True,
    "epm_persist_adapter_hf_repo": "superkaiba1/explore-persona-space",
}

sentinel = {
    "status": "${status}",
    "eval_numbers": eval_numbers,
    "eval_paths": [os.path.relpath(p, "${REPO_ROOT}") for p in eval_paths],
    "reproducibility_card": repro_card,
    "wandb_url": os.environ.get("WANDB_RUN_URL", ""),
    "hf_hub_url": "https://huggingface.co/superkaiba1/explore-persona-space",
    "worktree_path": "${REPO_ROOT}",
    "final_commit_sha": "${FINAL_COMMIT}",
    "gpu_hours_used": float(${gpu_hours_used}),
    "gpu_hours_budgeted": ${GPU_HOURS_BUDGETED},
    "plan_deviations": deviations,
    "started_at": "${START_TS}",
    "finished_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
}
print(json.dumps(sentinel, indent=2))
PYEOF
}

abort_with_failure() {
  # `abort_with_failure <reason>` — write sentinel status=failure + exit non-zero.
  local reason="$1"
  log_phase "failure" "abort" "reason=${reason}"
  record_deviation "abort" "${reason}"
  write_sentinel "failure" "0.0"
  exit 1
}

trap 'rc=$?; if [ $rc -ne 0 ]; then log_phase "trap" "exit" "rc=$rc"; fi' EXIT

# ─── 0a. Stage #411 wrong-claim panel from HF Hub (calibration source) ─────────
log_phase "stage_panel" "start"
if [ ! -f data/issue503/wrong_claims_en.jsonl ]; then
  uv run python <<'PYEOF'
import json, os
from pathlib import Path
from huggingface_hub import hf_hub_download

src = hf_hub_download(
    "superkaiba1/explore-persona-space-data",
    "issue411_sycophancy_cosine_gradient/data/wrong_claims/eval_50.jsonl",
    repo_type="dataset",
)
out_dir = Path("data/issue503")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "wrong_claims_en.jsonl"

# Source rows have keys: wrong_claim / correction / topic / topic_haiku.
# Calibration script expects a `claim` field; map wrong_claim → claim and add a
# stable id (claim_idx).
n_written = 0
with open(src) as f, open(out_path, "w") as g:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out_row = {
            "id": f"wc_{i:03d}",
            "claim": row["wrong_claim"],
            "correction": row.get("correction", ""),
            "topic": row.get("topic", row.get("topic_haiku", "")),
        }
        g.write(json.dumps(out_row) + "\n")
        n_written += 1
print(f"staged {n_written} rows to {out_path}")
PYEOF
  if [ $? -ne 0 ]; then
    abort_with_failure "stage_panel_failed"
  fi
fi
log_phase "stage_panel" "done" "rows=$(wc -l < data/issue503/wrong_claims_en.jsonl)"

# ─── 1. Phase 0.5: Bucket A judge calibration (ES + IT) ────────────────────────
log_phase "calibration" "start" "languages=es,it"

# Translate phase: Sonnet 4.5 paraphrase of the #411 panel into ES + IT.
uv run python scripts/issue503_judge_calibration.py \
    --phase translate \
    --languages es it \
    >> "${LOG_DIR}/issue-503-calibration.log" 2>&1
CALIB_TRANSLATE_RC=$?
if [ ${CALIB_TRANSLATE_RC} -ne 0 ]; then
  log_phase "calibration" "translate_failed" "rc=${CALIB_TRANSLATE_RC}"
  # Translation is best-effort: log + record deviation + continue. Bucket A may
  # be demoted by the sweep when it reads the (missing) kappa.json.
  record_deviation "calibration_translate_failed" "translate rc=${CALIB_TRANSLATE_RC}; Bucket A may be demoted"
else
  # Score + kappa phases run on the pod's vLLM after Qwen-7B generation; the
  # current bring-up runs translate only and lets the sweep read kappa.json if
  # present. Production score+kappa is a downstream artifact.
  uv run python scripts/issue503_judge_calibration.py \
      --phase all \
      --languages es it \
      >> "${LOG_DIR}/issue-503-calibration.log" 2>&1 || true
fi
log_phase "calibration" "done"

# ─── 2. Phase 0: #235 cross-lingual adapter retrain (conditional) ──────────────
log_phase "xling_retrain_check" "start"
XLING_FILES=$(uv run python -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('superkaiba1/explore-persona-space')
xling = [f for f in files if 'issue235_xling' in f.lower()]
print(len(xling))
" 2>/dev/null || echo "0")

if [ "${XLING_FILES}" -gt 0 ]; then
  log_phase "xling_retrain" "skip" "hf_adapters=${XLING_FILES}"
else
  log_phase "xling_retrain" "start" "cells=2 estimated_gpu_h=3"
  if [ -f scripts/issue503_xling_prep.py ]; then
    uv run python scripts/issue503_xling_prep.py \
        >> "${LOG_DIR}/issue-503-xling-retrain.log" 2>&1
    XLING_RC=$?
    if [ ${XLING_RC} -ne 0 ]; then
      # Bucket A xling cells depend on these adapters. Failed retrain → record
      # deviation, downgrade Bucket A to "partial coverage" at sentinel-write,
      # but continue with the rest of the sweep (B/C/D/E unaffected).
      log_phase "xling_retrain" "failed" "rc=${XLING_RC}"
      record_deviation "xling_retrain_failed" "issue503_xling_prep.py rc=${XLING_RC}; Bucket A xling cells skipped"
    else
      log_phase "xling_retrain" "done"
    fi
  else
    log_phase "xling_retrain" "skip" "reason=xling_prep_script_missing"
    record_deviation "xling_retrain_skipped" "scripts/issue503_xling_prep.py not present; Bucket A xling cells degraded"
  fi
fi

# ─── 2.5. Phase 0.7: Bucket D feature bundle + selection + 15 LoRA adapters ─────
# Round-6 GAP-5: the 5 He-et-al. selectors (D0_random / D1_representation /
# D2_gradient / D3_cosine / D4_format) × 3 seeds {0, 42, 137} = 15 adapters.
# Without these the sweep's Phase 1 cross_eval crashes the first time it sees
# a Bucket D source ("FileNotFoundError: adapter_config.json"). Per-cell
# fail-loud: a single training failure records a deviation and continues; the
# downstream cross_eval skips missing adapters via --skip-missing-adapter.

# Skip the whole phase if a previous run already uploaded all 15 adapters.
BUCKET_D_PRESENT=$(uv run python - <<'PYEOF' 2>/dev/null || echo "0"
from huggingface_hub import list_repo_files
try:
    files = list_repo_files("superkaiba1/explore-persona-space")
except Exception:
    print(0)
    raise SystemExit
sel = ("D0_random", "D1_representation", "D2_gradient", "D3_cosine", "D4_format")
n_present = 0
for s in sel:
    for seed in (0, 42, 137):
        prefix = f"issue503_bucket_d_{s}_seed{seed}/adapter/"
        if any(f.startswith(prefix) for f in files):
            n_present += 1
print(n_present)
PYEOF
)

if [ "${BUCKET_D_PRESENT:-0}" -ge 15 ]; then
  log_phase "bucket_d" "skip" "hf_adapters_present=${BUCKET_D_PRESENT}/15"
else
  log_phase "bucket_d_feature_bundle" "start"
  BUNDLE_DIR="${REPO_ROOT}/data/issue503/benign_data/feature_bundle"
  if [ ! -f "${BUNDLE_DIR}/grad_inner.npy" ]; then
    uv run python scripts/issue503_build_feature_bundle.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --layer 25 --position p5 \
        >> "${LOG_DIR}/issue-503-feature-bundle.log" 2>&1
    BUNDLE_RC=$?
  else
    log_phase "bucket_d_feature_bundle" "reuse" "dir=${BUNDLE_DIR}"
    BUNDLE_RC=0
  fi
  if [ ${BUNDLE_RC:-0} -ne 0 ]; then
    log_phase "bucket_d_feature_bundle" "failed" "rc=${BUNDLE_RC}"
    record_deviation "bucket_d_feature_bundle_failed" \
      "issue503_build_feature_bundle.py rc=${BUNDLE_RC}; all Bucket D cells skipped"
  else
    log_phase "bucket_d_feature_bundle" "done"

    log_phase "bucket_d_select" "start"
    uv run python scripts/issue503_benign_data_select.py \
        --feature-bundle "${BUNDLE_DIR}" \
        --top-k 100 \
        --method-independence \
        >> "${LOG_DIR}/issue-503-d-select.log" 2>&1
    SEL_RC=$?
    if [ ${SEL_RC} -ne 0 ]; then
      log_phase "bucket_d_select" "failed" "rc=${SEL_RC}"
      record_deviation "bucket_d_select_failed" \
        "issue503_benign_data_select.py rc=${SEL_RC}; all Bucket D cells skipped"
    else
      log_phase "bucket_d_select" "done"

      log_phase "bucket_d_train" "start" "selectors=5 seeds=3 total=15 estimated_gpu_h=15-30"
      for SEL in D0_random D1_representation D2_gradient D3_cosine D4_format; do
        for SEED in 0 42 137; do
          log_phase "bucket_d_train_cell" "start" "selector=${SEL} seed=${SEED}"
          SEL_JSONL="${REPO_ROOT}/eval_results/issue503/benign_data/${SEL}_seed${SEED}.jsonl"
          BENIGN_CORPUS="${BUNDLE_DIR}/benign_corpus.jsonl"
          SUBFOLDER="issue503_bucket_d_${SEL}_seed${SEED}/adapter"
          # Materialize the per-cell SFT JSONL (does NOT auto-launch training;
          # it prints the canonical command). We then launch train.py directly
          # with the EPM_PERSIST_ADAPTER env so the adapter uploads + verifies
          # to HF before local cleanup (per .claude/rules/upload-policy.md).
          uv run python scripts/issue503_benign_data_sft.py \
              --selector-jsonl "${SEL_JSONL}" \
              --benign-corpus "${BENIGN_CORPUS}" \
              --seed "${SEED}" \
              --selector-id "${SEL}" \
              --print-only \
              >> "${LOG_DIR}/issue-503-d-train-${SEL}-seed${SEED}.log" 2>&1
          MAT_RC=$?
          if [ ${MAT_RC} -ne 0 ]; then
            log_phase "bucket_d_train_cell" "failed" "selector=${SEL} seed=${SEED} stage=materialize rc=${MAT_RC}"
            record_deviation "bucket_d_train_cell_failed" \
              "selector=${SEL} seed=${SEED}: dataset materialize rc=${MAT_RC}"
            continue
          fi
          TRAIN_JSONL="${REPO_ROOT}/data/issue503/benign_data/${SEL}_seed${SEED}.jsonl"
          EPM_PERSIST_ADAPTER_HF_REPO="superkaiba1/explore-persona-space" \
          EPM_PERSIST_ADAPTER_SUBFOLDER="${SUBFOLDER}" \
          uv run python scripts/train.py \
              condition=issue503_benign_data_sft \
              +selector_id="${SEL}" \
              seed="${SEED}" \
              training.learning_rate=5e-5 \
              training.per_device_train_batch_size=1 \
              training.gradient_accumulation_steps=20 \
              training.num_train_epochs=5 \
              training.bf16=true \
              lora.r=32 lora.lora_alpha=256 \
              data.training_jsonl="${TRAIN_JSONL}" \
              >> "${LOG_DIR}/issue-503-d-train-${SEL}-seed${SEED}.log" 2>&1
          TRAIN_RC=$?
          if [ ${TRAIN_RC} -ne 0 ]; then
            log_phase "bucket_d_train_cell" "failed" "selector=${SEL} seed=${SEED} stage=train rc=${TRAIN_RC}"
            record_deviation "bucket_d_train_cell_failed" \
              "selector=${SEL} seed=${SEED}: train.py rc=${TRAIN_RC}"
          else
            # Fail-loud verify per upload-policy.md: confirm the adapter
            # actually landed before the next cell wipes the checkpoint dir.
            VERIFY=$(uv run python -c "
from huggingface_hub import list_repo_files
fs = list_repo_files('superkaiba1/explore-persona-space')
n = sum(1 for f in fs if f.startswith('${SUBFOLDER}/'))
print(n)
" 2>/dev/null || echo "0")
            if [ "${VERIFY:-0}" -lt 1 ]; then
              log_phase "bucket_d_train_cell" "upload_unverified" "selector=${SEL} seed=${SEED}"
              record_deviation "bucket_d_train_cell_upload_unverified" \
                "selector=${SEL} seed=${SEED}: train rc=0 but HF Hub has no ${SUBFOLDER}/* files"
            else
              log_phase "bucket_d_train_cell" "done" "selector=${SEL} seed=${SEED} hf_files=${VERIFY}"
            fi
          fi
        done
      done
      log_phase "bucket_d_train" "done"
    fi
  fi
fi

# ─── 2.6. Phase 0.8: broad-syco source training (compliment → general) ──────────
# Plan §3.2.2 flags this LOW confidence (never validated in-house). Seed=0 first
# as a smoke gate; if seed=0 verify-on-HF fails OR upload missing, drop seed=137
# and record a deviation. Downstream cross_eval will --skip-missing-adapter on
# the broad-syco source for any B→B cells that wanted it.

BROAD_SYCO_PRESENT=$(uv run python - <<'PYEOF' 2>/dev/null || echo "0"
from huggingface_hub import list_repo_files
try:
    files = list_repo_files("superkaiba1/explore-persona-space")
except Exception:
    print(0)
    raise SystemExit
n = 0
for seed in (0, 137):
    prefix = f"issue503_broad_syco_seed{seed}/adapter/"
    if any(f.startswith(prefix) for f in files):
        n += 1
print(n)
PYEOF
)

if [ "${BROAD_SYCO_PRESENT:-0}" -ge 2 ]; then
  log_phase "broad_syco_source" "skip" "hf_adapters_present=${BROAD_SYCO_PRESENT}/2"
else
  log_phase "broad_syco_source_build" "start"
  uv run python scripts/issue503_build_broad_syco_dataset.py --seeds 0 137 \
      >> "${LOG_DIR}/issue-503-broad-syco-build.log" 2>&1
  BUILD_RC=$?
  if [ ${BUILD_RC} -ne 0 ]; then
    log_phase "broad_syco_source_build" "failed" "rc=${BUILD_RC}"
    record_deviation "broad_syco_source_build_failed" \
      "issue503_build_broad_syco_dataset.py rc=${BUILD_RC}; B→B broad-syco cells skipped"
  else
    log_phase "broad_syco_source_build" "done"
    log_phase "broad_syco_source_train" "start" "seeds=2 estimated_gpu_h=2-4"
    BROAD_SYCO_SEED0_OK=false
    for SEED in 0 137; do
      log_phase "broad_syco_source_train_cell" "start" "seed=${SEED}"
      SUBFOLDER="issue503_broad_syco_seed${SEED}/adapter"
      EPM_PERSIST_ADAPTER_HF_REPO="superkaiba1/explore-persona-space" \
      EPM_PERSIST_ADAPTER_SUBFOLDER="${SUBFOLDER}" \
      uv run python scripts/train.py \
          condition=issue503_broad_syco_source \
          seed="${SEED}" \
          >> "${LOG_DIR}/issue-503-broad-syco-train-seed${SEED}.log" 2>&1
      TRAIN_RC=$?
      if [ ${TRAIN_RC} -ne 0 ]; then
        log_phase "broad_syco_source_train_cell" "failed" "seed=${SEED} rc=${TRAIN_RC}"
        record_deviation "broad_syco_source_train_failed" \
          "seed=${SEED}: train.py rc=${TRAIN_RC}; B→B broad-syco cells degraded"
        if [ "${SEED}" = "0" ]; then
          # Smoke gate: if seed=0 failed, drop seed=137 too (LOW-confidence
          # recipe — plan §12 #6).
          log_phase "broad_syco_source_train" "abort_seed137" "reason=seed0_failed"
          record_deviation "broad_syco_source_seed137_dropped" \
            "seed=0 failed; dropping seed=137 per plan §12 #6 smoke-gate policy"
          break
        fi
      else
        VERIFY=$(uv run python -c "
from huggingface_hub import list_repo_files
fs = list_repo_files('superkaiba1/explore-persona-space')
n = sum(1 for f in fs if f.startswith('${SUBFOLDER}/'))
print(n)
" 2>/dev/null || echo "0")
        if [ "${VERIFY:-0}" -lt 1 ]; then
          log_phase "broad_syco_source_train_cell" "upload_unverified" "seed=${SEED}"
          record_deviation "broad_syco_source_upload_unverified" \
            "seed=${SEED}: train rc=0 but HF Hub has no ${SUBFOLDER}/* files"
        else
          log_phase "broad_syco_source_train_cell" "done" "seed=${SEED} hf_files=${VERIFY}"
          if [ "${SEED}" = "0" ]; then
            BROAD_SYCO_SEED0_OK=true
          fi
        fi
      fi
    done
    log_phase "broad_syco_source_train" "done" "seed0_ok=${BROAD_SYCO_SEED0_OK}"
  fi
fi

# ─── 3. Pod-side end-to-end smoke for Buckets A / D / E ────────────────────────
for BUCKET in A D E; do
  log_phase "smoke" "start" "bucket=${BUCKET}"
  uv run python scripts/issue503_pod_smoke.py \
      --bucket "${BUCKET}" \
      --max-prompts 8 \
      --seeds 0 \
      >> "${LOG_DIR}/issue-503-smoke-${BUCKET}.log" 2>&1
  SMOKE_RC=$?
  if [ ${SMOKE_RC} -ne 0 ]; then
    log_phase "smoke" "failed" "bucket=${BUCKET} rc=${SMOKE_RC}"
    # Smoke failures DO abort the production launch — they exist to catch
    # config/architectural drift before we burn 170 GPU-h.
    abort_with_failure "smoke_bucket_${BUCKET}_failed_rc_${SMOKE_RC}"
  fi
  log_phase "smoke" "done" "bucket=${BUCKET}"
done

# ─── 4. Production sweep: --all-cells × seeds [0, 137] ─────────────────────────
log_phase "production" "start" "cells=135 seeds=0,137 budget_gpu_h=${GPU_HOURS_BUDGETED}"
PROD_START_EPOCH=$(date +%s)

uv run python scripts/issue503_sweep.py \
    --all-cells \
    --seeds 0 137 \
    --skip-missing-adapter \
    >> "${LOG_DIR}/issue-503-production.log" 2>&1
PROD_RC=$?

PROD_END_EPOCH=$(date +%s)
PROD_WALL_S=$((PROD_END_EPOCH - PROD_START_EPOCH))
# 4 H100s active → wall-clock × 4 ≈ GPU-h (rough; the sweep itself logs precise
# per-cell timings via WandB).
GPU_HOURS_USED=$(python3 -c "print(round(${PROD_WALL_S} * 4 / 3600.0, 2))")

if [ ${PROD_RC} -ne 0 ]; then
  log_phase "production" "failed" "rc=${PROD_RC} gpu_h_used=${GPU_HOURS_USED}"
  record_deviation "production_sweep_failed" "issue503_sweep.py rc=${PROD_RC} after ${GPU_HOURS_USED} GPU-h"
  write_sentinel "failure" "${GPU_HOURS_USED}"
  exit ${PROD_RC}
fi

log_phase "production" "done" "gpu_h_used=${GPU_HOURS_USED}"

# ─── 5. Terminal sentinel ──────────────────────────────────────────────────────
write_sentinel "done" "${GPU_HOURS_USED}"
log_phase "done" "ok" "sentinel=${SENTINEL}"
exit 0
