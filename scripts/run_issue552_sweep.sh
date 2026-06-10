#!/usr/bin/env bash
# Issue #552 — pod-side end-to-end driver: benign plain-SFT control arm for
# the #521 shift-geometry contrast.
#
# Plan §4.2. Single command the experimenter launches under nohup; chains:
#
#   Step 0   env preamble (asserts OPENAI_API_KEY / HF_TOKEN / WANDB_API_KEY
#            non-empty — preflight does NOT check OPENAI_API_KEY, so fail
#            loud pre-GPU rather than mid-gate).
#   Step 1   good-corpus prep (hard assert 5,899 rows) + training-mix HF upload.
#   Step 2   2-step recipe smoke train + recipe-assert verifier (HALT gate).
#   Step 3   3-seed production train (parallel, 1 GPU each) with per-cell
#            EPM_PERSIST_ADAPTER_* + HF persist verify (#458 pattern).
#   Step 4   INVERTED EM-installation gate (canonical 8x100 surface,
#            gpt-4o-2024-08-06; PASS iff every cell L <= 0.05). A cell > 5%
#            HALTS geometry — recorded in the results sentinel as its own
#            finding (plan §7 gate 2), NOT a crash.
#   Step 4b  gate raw-completion upload to HF + merged-dir cleanup (~45 GB).
#   Step 5   adapter staging from HF -> eval_results/issue_552/benign_seed{S}/adapter.
#   Step 6   pipeline smoke = the PRODUCTION dispatcher restricted to one cell
#            (--arms benign --seeds 1 --variants same) + schema asserts.
#   Step 7   Phase C production (9 cells = 3 variants x 3 seeds, --save-per-question).
#   Step 7b  TENSOR UPLOAD CHECKPOINT: shifts/*.pt -> HF analysis_tensors/,
#            fail-loud list_repo_files verify BEFORE Phase D (the #521
#            lost-tensor incident fix; pod may not terminate without it).
#   Step 8   Phase D (SVD + dual nulls, on-pod CPU, <15 min) + 9-file assert.
#   sentinel + [phase=done]  poll_pipeline.py contract
#            (/workspace/logs/issue-552-epm_results-<epoch>.json).
#
# Run via the experimenter under nohup. This script NEVER shells out to
# scripts/task.py (CLAUDE.md "Pod-side code NEVER shells out to task.py");
# all marker posting happens VM-side via the sentinel + poll_pipeline.py.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# ── HF cache must live on the MooseFS volume, not /root (CLAUDE.md). ──
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"

# ──────────────────────────────────────────────────────────────────────
# Step 0 — env preamble: load .env if present, then HARD-assert the keys
# every downstream phase needs. preflight checks WANDB/HF/ANTHROPIC but
# NOT OPENAI_API_KEY (plan §12 assumption 17) — the inverted gate's
# gpt-4o judge dies hours in without it, so assert BEFORE any GPU spend.
# ──────────────────────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
for key in OPENAI_API_KEY HF_TOKEN WANDB_API_KEY; do
  if [[ -z "${!key:-}" ]]; then
    echo "[phase=fail] $(date -Is) env preamble: required env var $key is empty/unset" >&2
    echo "Provision/bootstrap must push .env (bootstrap_pod.sh does); refusing to start." >&2
    exit 11
  fi
done
echo "[phase=env_ok] $(date -Is) OPENAI_API_KEY/HF_TOKEN/WANDB_API_KEY all non-empty"

# Smoke hook: assert env + exit cleanly without touching data/GPU.
if [[ "${EPM_CHECK_ENV_ONLY:-0}" == "1" ]]; then
  echo "[phase=done] EPM_CHECK_ENV_ONLY=1 — env preamble PASS, exiting"
  exit 0
fi

# ── Durable-checkpoint policy per #458/#521 (closes the silent-loss hole).
# EPM_PERSIST_ADAPTER_* are scoped to the per-cell subshells ONLY (the
# smoke train runs with neither set); drop any inherited values first.
unset EPM_PERSIST_ADAPTER_HF_REPO EPM_PERSIST_ADAPTER_SUBFOLDER
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

SEEDS=(42 137 256)
N_GPUS="${N_GPUS:-4}"
MAX_STEPS="${MAX_STEPS:-375}"     # plan §11 (#458 fixed-step design via #521).
SMOKE_SEED="${SMOKE_SEED:-42}"
SMOKE_GPU="${SMOKE_GPU:-0}"
COND="issue404_pair_turner_good_medical"
OUT_DIR="eval_results/issue_552"
GATE_BASE="$OUT_DIR/em_rate_gate_firstplot"
HF_DATA_REPO="superkaiba1/explore-persona-space-data"
HF_MODEL_REPO="superkaiba1/explore-persona-space"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue-552}"
mkdir -p "$LOG_DIR" /workspace/logs "$OUT_DIR"

phase() {
  echo "[phase=$1] $(date -Is) $2"
}

fail_loud() {
  local rc="$1"; shift
  local reason="$*"
  phase fail "rc=$rc reason=$reason"
  exit "$rc"
}

# ──────────────────────────────────────────────────────────────────────
# Step 1 — good-corpus prep (hard 5,899-row parity assert) + HF upload
# ──────────────────────────────────────────────────────────────────────
phase prep_good_corpus "pulling good_medical_advice_6k.jsonl + parent slice protocol + HF upload"
uv run python scripts/issue_552_prep_good_corpus.py 2>&1 | tee "$LOG_DIR/prep_good_corpus.log" || \
  fail_loud "$?" "prep_good_corpus_failed (row-parity, pool-overlap, or HF upload)"

# ──────────────────────────────────────────────────────────────────────
# Step 2 — recipe smoke train + verifier (METHODOLOGY HALT GATE, plan §7 gate 1)
# ──────────────────────────────────────────────────────────────────────
phase recipe_smoke "2-step no-upload smoke train + recipe assert (seed=${SMOKE_SEED} gpu=${SMOKE_GPU})"
SMOKE_LOG="$LOG_DIR/recipe_smoke_seed${SMOKE_SEED}.log"
uv run python scripts/train.py \
  condition="$COND" \
  training=turner_em lora=turner_em \
  +training.max_steps=2 \
  seed="$SMOKE_SEED" +gpu_id="$SMOKE_GPU" \
  upload_to=none \
  2>&1 | tee "$SMOKE_LOG" || fail_loud "$?" "recipe_smoke_train_failed"

phase recipe_assert "asserting run_result.json records all 14 turner_em recipe values verbatim"
uv run python scripts/issue_521_em_recipe_smoke.py --seed "$SMOKE_SEED" --condition "$COND" \
  2>&1 | tee "$LOG_DIR/recipe_assert.log" || \
  fail_loud "$?" "recipe_smoke_assert_failed_DO_NOT_PROCEED_TO_PRODUCTION"

# ──────────────────────────────────────────────────────────────────────
# Step 3 — 3-seed production train (parallel, 1 GPU each)
# ──────────────────────────────────────────────────────────────────────
phase prod_train "launching 3-seed parallel train (GPUs 0/1/2; GPU 3 idle)"

pids=()
for i in 0 1 2; do
  SEED="${SEEDS[$i]}"
  GPU="$i"
  RUN_NAME="${COND}_seed${SEED}"
  MERGED_DIR="$REPO_ROOT/models/${RUN_NAME}/sft_narrow_merged"
  CELL_LOG="$LOG_DIR/prod_train_seed${SEED}.log"

  (
    set -euo pipefail
    # Per-cell durable adapter destination (#458/#521 pattern): both env
    # vars exported TOGETHER, scoped to this subshell only.
    export EPM_PERSIST_ADAPTER_HF_REPO="$HF_MODEL_REPO"
    export EPM_PERSIST_ADAPTER_SUBFOLDER="adapters/issue_552/benign_turner_seed${SEED}"
    phase cell_train "GPU=$GPU seed=$SEED max_steps=$MAX_STEPS"
    uv run python scripts/train.py \
      condition="$COND" \
      training=turner_em lora=turner_em \
      +training.max_steps="$MAX_STEPS" \
      seed="$SEED" +gpu_id="$GPU" \
      upload_to=none \
      >> "$CELL_LOG" 2>&1

    # FAIL LOUD if the run did not produce the expected merged dir (the
    # inverted gate loads it via EPM_ISSUE404_LOCAL_MERGED_BASE).
    if [[ ! -f "$MERGED_DIR/config.json" ]]; then
      phase cell_train_fail "seed=$SEED merged dir missing at $MERGED_DIR"
      exit 17
    fi

    # Persist-check: verify the adapter actually landed on HF.
    uv run python - <<PY
from huggingface_hub import list_repo_files
prefix = "adapters/issue_552/benign_turner_seed${SEED}"
files = list_repo_files("$HF_MODEL_REPO", repo_type="model", revision="main")
matched = [f for f in files if f.startswith(prefix)]
assert any("adapter_config.json" in f for f in matched), \
    f"adapter_config.json missing on HF under {prefix}: {matched}"
assert any("adapter_model.safetensors" in f for f in matched), \
    f"adapter_model.safetensors missing on HF under {prefix}: {matched}"
print(f"HF adapter persist PASS: seed=${SEED}, {len(matched)} files at {prefix}/")
PY
  ) &
  pids+=($!)
done

fail_any=0
for pid in "${pids[@]}"; do
  rc=0
  wait "$pid" || rc=$?
  if (( rc != 0 )); then
    phase cell_subprocess_failed "pid=$pid rc=$rc"
    fail_any=1
  fi
done
if (( fail_any )); then
  fail_loud 18 "one_or_more_prod_train_subprocess_failed"
fi

# ──────────────────────────────────────────────────────────────────────
# Step 4 — INVERTED EM-installation gate (plan §7 gate 2; BEFORE any geometry)
# ──────────────────────────────────────────────────────────────────────
phase em_gate_inverted "canonical 8x100 gate on 3 benign cells (gpt-4o-2024-08-06); PASS iff all L <= 0.05"
GATE_RC=0
EPM_ISSUE404_LOCAL_MERGED_BASE="$REPO_ROOT/models" \
uv run python scripts/issue404_outcome_eval.py \
  --pairs turner_good_medical --seeds "${SEEDS[@]}" \
  --skip-calibration --judge-model gpt-4o-2024-08-06 \
  --output-base "$GATE_BASE" \
  2>&1 | tee "$LOG_DIR/em_gate_inverted.log" || GATE_RC=$?
if (( GATE_RC != 0 )); then
  fail_loud "$GATE_RC" "issue404_outcome_eval_failed (judge/vLLM error, NOT a gate decision)"
fi

phase em_gate_decision "applying the INVERTED decision rule to the 3 outcome JSONs"
GATE_DECISION_RC=0
uv run python scripts/issue552_gate_decision.py \
  --outcome-dir "$GATE_BASE/outcome" --seeds "${SEEDS[@]}" \
  2>&1 | tee "$LOG_DIR/em_gate_decision.log" || GATE_DECISION_RC=$?
if (( GATE_DECISION_RC != 0 && GATE_DECISION_RC != 21 )); then
  fail_loud "$GATE_DECISION_RC" "gate_decision_script_errored (not a gate FAIL)"
fi

# ──────────────────────────────────────────────────────────────────────
# Step 4b — gate raw-completion upload (BEFORE merged-dir cleanup; upload
# policy: raw completions MUST reach HF before pod termination), then
# delete the 3 x ~15 GB merged dirs (MooseFS quota headroom).
# ──────────────────────────────────────────────────────────────────────
phase gate_raw_upload "uploading gate raw completions + judge scores to HF (single bulk commit)"
uv run python - <<'PY'
"""Bulk-upload the inverted gate's raw artifacts + verify the listing."""
from pathlib import Path

from huggingface_hub import list_repo_files, upload_folder

repo = "superkaiba1/explore-persona-space-data"
src = Path("eval_results/issue_552/em_rate_gate_firstplot/outcome")
expected = sorted(p.name for p in src.glob("raw_completions_*.json"))
assert len(expected) == 3, f"expected 3 raw_completions files, found {expected}"
upload_folder(
    folder_path=str(src),
    path_in_repo="issue552_benign_control/em_rate_gate_firstplot/raw_completions",
    repo_id=repo,
    repo_type="dataset",
    allow_patterns=["raw_completions_*.json", "judge_scores_*.json"],
    commit_message="#552: inverted-gate raw completions + judge scores (3 benign cells)",
)
files = list_repo_files(repo, repo_type="dataset", revision="main")
prefix = "issue552_benign_control/em_rate_gate_firstplot/raw_completions/"
landed = {f[len(prefix):] for f in files if f.startswith(prefix)}
missing = [n for n in expected if n not in landed]
assert not missing, f"raw-completion upload verification FAILED; missing on HF: {missing}"
print(f"raw-completion upload verified: {sorted(landed)}")
PY

phase merged_cleanup "deleting 3 merged dirs (~45 GB) — adapters are the persisted artifact"
for SEED in "${SEEDS[@]}"; do
  rm -rf "$REPO_ROOT/models/${COND}_seed${SEED}/sft_narrow_merged"
done

# ── Gate-FAIL branch: halt geometry, record the finding, exit CLEANLY.
if (( GATE_DECISION_RC == 21 )); then
  phase halt_geometry_em_gate "inverted gate FAILED (a benign cell > 5%); geometry forgone by design"
  uv run python scripts/issue552_write_sentinel.py --mode gate_halt \
    2>&1 | tee "$LOG_DIR/sentinel_gate_halt.log" || \
    fail_loud "$?" "gate_halt_sentinel_write_failed"
  phase done "issue-552 complete (geometry halted by inverted gate — finding recorded in sentinel)"
  exit 0
fi

# ──────────────────────────────────────────────────────────────────────
# Step 5 — adapter staging: HF -> eval_results/issue_552/benign_seed{S}/adapter
# (matches the dispatcher's output_dir/{arm}_seed{S}/adapter convention)
# ──────────────────────────────────────────────────────────────────────
phase stage_benign_adapters "staging benign adapters from HF to local layout"
uv run python scripts/issue_521_stage_em_turner_adapters.py \
  --output-dir "$OUT_DIR" \
  --seeds "${SEEDS[@]}" \
  --hf-prefix-template "adapters/issue_552/benign_turner_seed{seed}" \
  --cell-template "benign_seed{seed}" \
  2>&1 | tee "$LOG_DIR/stage_benign_adapters.log" || \
  fail_loud "$?" "stage_benign_adapters_failed"

# ──────────────────────────────────────────────────────────────────────
# Step 6 — pipeline smoke = the PRODUCTION dispatcher, one cell
# (architectural parity: smoke IS sweep with --arms benign --seeds 1)
# ──────────────────────────────────────────────────────────────────────
phase pipeline_smoke "one-cell smoke through the production dispatcher (benign seed42, same variant)"
uv run python scripts/issue_519_dispatch.py --mode sweep \
  --arms benign --seeds 1 --variants same \
  --save-per-question \
  --skip-phase a1 a23 b0_smoke b e \
  --output-dir "$OUT_DIR" \
  --personas-json eval_results/issue_521/inputs/personas.json \
  --questions-json eval_results/issue_521/inputs/questions.json \
  --base-cosines-json eval_results/issue_521/inputs/base_cosines.json \
  --n-gpus 1 \
  2>&1 | tee "$LOG_DIR/pipeline_smoke.log" || fail_loud "$?" "pipeline_smoke_failed"

phase pipeline_smoke_assert "asserting smoke artifacts + per-cell JSON schema"
uv run python - <<'PY'
"""Smoke-cell asserts (plan §4.2 Step 6): .pt + per-question tensor + SVD JSON schema."""
import json
from pathlib import Path

import torch

pt = Path("eval_results/issue_552/shifts/same_benign_seed42.pt")
assert pt.exists(), f"smoke shift tensor missing: {pt}"
payload = torch.load(pt, map_location="cpu", weights_only=False)
shifts = payload["shifts"]
assert len(shifts) == 14, f"expected 14 personas, got {len(shifts)}"
for p_name, entry in shifts.items():
    assert entry["delta_v"].shape == (3584,), (p_name, entry["delta_v"].shape)
    assert "delta_v_per_question" in entry, f"per-question tensor missing for {p_name}"
    pq = entry["delta_v_per_question"]
    assert pq.dim() == 2 and pq.shape[1] == 3584, (p_name, pq.shape)
    assert "delta_v_mean_resp" in entry, f"mean-over-response read missing for {p_name}"

svd = Path("eval_results/issue_552/svd/same_benign_seed42.json")
assert svd.exists(), f"smoke SVD JSON missing: {svd}"
d = json.loads(svd.read_text())
assert len(d["U1"]) == 3584, len(d["U1"])
assert len(d["singular_values"]) == 14, len(d["singular_values"])
for key in ("s_top1_frac", "mean_cos_to_U1", "cos_to_U1", "sign_flip_p95", "row_shuffle_p95"):
    assert key in d, f"per-cell SVD JSON missing key {key!r} (parent schema mismatch)"
print("pipeline smoke asserts PASS: tensor + per-question + SVD schema all match the parent contract")
PY

# ──────────────────────────────────────────────────────────────────────
# Step 7 — Phase C production: 9 cells (3 variants x 3 seeds), per-question on
# (Phase D EXCLUDED here so the tensor-upload checkpoint sits between C and D)
# ──────────────────────────────────────────────────────────────────────
phase phase_c_production "Phase C: 9 cells (3 variants x 3 seeds) with --save-per-question"
uv run python scripts/issue_519_dispatch.py --mode sweep \
  --arms benign --seeds 3 --variants same base on_policy \
  --save-per-question \
  --skip-phase a1 a23 b0_smoke b e d \
  --output-dir "$OUT_DIR" \
  --personas-json eval_results/issue_521/inputs/personas.json \
  --questions-json eval_results/issue_521/inputs/questions.json \
  --base-cosines-json eval_results/issue_521/inputs/base_cosines.json \
  --n-gpus "$N_GPUS" \
  2>&1 | tee "$LOG_DIR/phase_c_production.log" || fail_loud "$?" "phase_c_production_failed"

# ──────────────────────────────────────────────────────────────────────
# Step 7b — TENSOR UPLOAD CHECKPOINT (the #521 incident fix): shifts/*.pt
# (means + per-question) -> HF analysis_tensors/, verified BEFORE Phase D.
# ──────────────────────────────────────────────────────────────────────
phase tensor_upload_checkpoint "uploading shifts/*.pt to HF + fail-loud verify BEFORE Phase D"
uv run python - <<'PY'
"""Upload all shift tensors in ONE bulk commit (HF 256-commits/hr rule), then
fail-loud verify every expected file via list_repo_files."""
from pathlib import Path

from huggingface_hub import list_repo_files, upload_folder

repo = "superkaiba1/explore-persona-space-data"
src = Path("eval_results/issue_552/shifts")
expected = sorted(p.name for p in src.glob("*"))
pt_files = [n for n in expected if n.endswith(".pt")]
# 9 production cells (3 variants x 3 seeds); the smoke cell re-ran inside them.
assert len(pt_files) >= 9, f"expected >=9 shift .pt files before upload, found {pt_files}"
upload_folder(
    folder_path=str(src),
    path_in_repo="issue552_benign_control/analysis_tensors",
    repo_id=repo,
    repo_type="dataset",
    commit_message="#552: benign-arm shift tensors (means + per-question) — pre-Phase-D checkpoint",
)
files = list_repo_files(repo, repo_type="dataset", revision="main")
prefix = "issue552_benign_control/analysis_tensors/"
landed = {f[len(prefix):] for f in files if f.startswith(prefix)}
missing = [n for n in expected if n not in landed]
assert not missing, (
    f"TENSOR UPLOAD VERIFICATION FAILED — missing on HF: {missing}. "
    f"Refusing to proceed to Phase D (the #521 lost-tensor incident fix: "
    f"the pod may not terminate without these on HF)."
)
print(f"tensor upload checkpoint PASS: {len(landed)} files verified under {prefix}")
PY

# ──────────────────────────────────────────────────────────────────────
# Step 8 — Phase D: SVD + dual nulls (1,000 reps) per (variant, seed); CPU, <15 min
# ──────────────────────────────────────────────────────────────────────
phase phase_d_svd "Phase D: SVD + row-shuffle/sign-flip nulls over the 9 benign cells"
uv run python scripts/issue_519_dispatch.py --mode sweep \
  --arms benign --seeds 3 --variants same base on_policy \
  --skip-phase a1 a23 b0_smoke b c e \
  --output-dir "$OUT_DIR" \
  --personas-json eval_results/issue_521/inputs/personas.json \
  --questions-json eval_results/issue_521/inputs/questions.json \
  --base-cosines-json eval_results/issue_521/inputs/base_cosines.json \
  --n-gpus "$N_GPUS" \
  2>&1 | tee "$LOG_DIR/phase_d_svd.log" || fail_loud "$?" "phase_d_svd_failed"

phase phase_d_assert "asserting all 9 per-cell SVD JSONs exist"
uv run python - <<'PY'
from pathlib import Path

svd_dir = Path("eval_results/issue_552/svd")
missing = []
for variant in ("same", "base", "on_policy"):
    for seed in (42, 137, 256):
        p = svd_dir / f"{variant}_benign_seed{seed}.json"
        if not p.exists():
            missing.append(str(p))
assert not missing, f"Phase D output incomplete; missing per-cell SVD JSONs: {missing}"
print("Phase D assert PASS: all 9 per-cell SVD JSONs present")
PY

# ──────────────────────────────────────────────────────────────────────
# End-of-run sentinel (poll_pipeline.py contract: _SENTINEL_REQUIRED_KEYS)
# ──────────────────────────────────────────────────────────────────────
phase write_sentinel "writing end-of-run results sentinel"
uv run python scripts/issue552_write_sentinel.py --mode done \
  2>&1 | tee "$LOG_DIR/sentinel_done.log" || \
  fail_loud "$?" "results_sentinel_write_failed"

# poll_pipeline.py declares status="done" ONLY when the most recent
# [phase=...] line is [phase=done] — emit it AFTER the sentinel write.
phase done "issue-552 benign-control sweep complete"
