#!/usr/bin/env bash
# Issue #552 — RECOVERY resume driver (round 2): re-enters the benign
# plain-SFT control-arm pipeline AFTER the round-1 HF-quota crash.
#
# Context: the round-1 sweep (scripts/run_issue552_sweep.sh) crashed at the
# prod_train join (rc=18) ONLY because HF Hub returned 403 "exceeded your
# public storage space" on the per-seed adapter persist. Training itself
# SUCCEEDED: all 3 seeds (42/137/256) completed 375/375 steps and each
# models/issue404_pair_turner_good_medical_seed{S}/ dir on the pod carries
# BOTH sft_narrow_adapter/ AND sft_narrow_merged/. Steps 1-3 of the sweep
# (corpus prep + HF upload, recipe smoke + assert, production train) are
# DONE and MUST NOT re-run. Adapter durability is handled OUTSIDE this
# script (VM backup at eval_results/issue_552/adapter_backup/ + WandB
# Artifact); this script makes NO adapter upload to HF.
#
# What runs here (plan §4.2 Steps 4 → end, copied from the sweep script):
#
#   Step 0   env preamble (same asserts as the sweep).
#   Step R   resume preflight (NEW): assert the round-1 train artifacts
#            exist on the pod per seed — merged config.json + tokenizer
#            files + weight shards (index-validated when sharded), and
#            adapter safetensors/config; distinct rc per missing-artifact
#            class. Covers the FULL file set the gate's local-mode vLLM
#            load consumes (round-3 fix, concern
#            resume-preflight-merged-artifact-coverage).
#   Step 4   INVERTED EM-installation gate (canonical 8x100 surface,
#            gpt-4o-2024-08-06; PASS iff every cell L <= 0.05).
#   Step 4b  gate raw-completion upload to HF + merged-dir cleanup (~45 GB).
#            (Small JSON uploads to the DATASET repo still pass under the
#            quota — verified empirically 2026-06-10; only >=30MB LFS 403s.)
#   Step 5   adapter staging — THE ONE SUBSTANTIVE CHANGE vs the sweep:
#            LOCAL copy models/<cond>_seed{S}/sft_narrow_adapter/ ->
#            eval_results/issue_552/benign_seed{S}/adapter/ (excluding
#            checkpoint-* subdirs), replacing the HF-download stager
#            (issue_521_stage_em_turner_adapters.py) which cannot work
#            while the model repo is over quota. Mirrors the stager's
#            destination layout + required-file/size asserts.
#   Step 6   pipeline smoke = the PRODUCTION dispatcher restricted to one
#            cell (--arms benign --seeds 1 --variants same) + schema asserts.
#   Step 7   Phase C production (9 cells = 3 variants x 3 seeds).
#   Step 7b  TENSOR UPLOAD CHECKPOINT: shifts/*.pt -> HF analysis_tensors/.
#   Step 8   Phase D (SVD + dual nulls) + 9-file assert.
#   sentinel + [phase=done]  poll_pipeline.py contract.
#
# Smoke hooks (VM-side, no GPU):
#   EPM_CHECK_ENV_ONLY=1      env preamble only, then clean exit.
#   EPM_RESUME_STAGE_ONLY=1   resume preflight + local staging, then clean
#                             exit (pair with REPO_ROOT/MODELS_ROOT/
#                             POD_LOG_DIR pointed at a fixture tree).
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

# ── NO adapter HF persist this round: the model repo is over its public
# storage quota (the round-1 403). Unset defensively so nothing inherited
# from the environment re-triggers an upload path; no training runs here
# anyway, so these are belt-and-braces only.
unset EPM_PERSIST_ADAPTER_HF_REPO EPM_PERSIST_ADAPTER_SUBFOLDER
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

SEEDS=(42 137 256)
N_GPUS="${N_GPUS:-4}"
COND="issue404_pair_turner_good_medical"
OUT_DIR="eval_results/issue_552"
GATE_BASE="$OUT_DIR/em_rate_gate_firstplot"
# Round-1 train artifacts root (override only for the VM fixture smoke).
MODELS_ROOT="${MODELS_ROOT:-$REPO_ROOT/models}"
# Sentinel landing dir pre-create (override only for the VM fixture smoke;
# issue552_write_sentinel.py defaults to /workspace/logs independently).
POD_LOG_DIR="${POD_LOG_DIR:-/workspace/logs}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue-552}"
mkdir -p "$LOG_DIR" "$POD_LOG_DIR" "$OUT_DIR"

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
# Step 5 body — LOCAL adapter staging (replaces the HF-download stager).
# Mirrors issue_521_stage_em_turner_adapters.py::_stage_one semantics:
#   - destination layout $OUT_DIR/benign_seed{S}/adapter/ (what the
#     issue_519_dispatch.py production dispatcher loads, line ~469);
#   - skip-if-already-staged with stale/partial detection;
#   - REQUIRED files adapter_config.json + adapter_model.safetensors;
#   - size sanity (config >= 100 B, safetensors >= 1024 B).
# checkpoint-* subdirs (mid-train Trainer checkpoints) are EXCLUDED —
# the dispatcher loads the final adapter files only.
# Defined as a function so the EPM_RESUME_STAGE_ONLY smoke hook and the
# production Step 5 slot share one implementation.
# ──────────────────────────────────────────────────────────────────────
stage_benign_adapters_local() {
  phase stage_benign_adapters "staging benign adapters via LOCAL copy (HF model repo over quota — no HF round-trip)"
  local SEED SRC DEST CFG SFT cfg_sz sft_sz
  for SEED in "${SEEDS[@]}"; do
    SRC="$MODELS_ROOT/${COND}_seed${SEED}/sft_narrow_adapter"
    DEST="$OUT_DIR/benign_seed${SEED}/adapter"
    CFG="$DEST/adapter_config.json"
    SFT="$DEST/adapter_model.safetensors"
    if [[ -f "$CFG" && -f "$SFT" ]]; then
      cfg_sz=$(stat -c%s "$CFG")
      sft_sz=$(stat -c%s "$SFT")
      if (( cfg_sz >= 100 && sft_sz >= 1024 )); then
        phase stage_skip "benign_seed${SEED} already staged at $DEST (cfg=${cfg_sz}B, safetensors=${sft_sz}B)"
        continue
      fi
      fail_loud 34 "stage_benign_adapters: pre-existing staged adapter at $DEST is suspiciously small (cfg=${cfg_sz}B, safetensors=${sft_sz}B) — partial previous stage; delete $DEST and re-run"
    fi
    mkdir -p "$DEST"
    # Top-level copy excluding checkpoint-* subdirs. The `+` exec form
    # propagates cp failures into find's exit status (set -e catches).
    find "$SRC" -mindepth 1 -maxdepth 1 ! -name 'checkpoint-*' -exec cp -r -t "$DEST" {} +
    [[ -f "$SFT" ]] || fail_loud 35 "stage_benign_adapters: adapter_model.safetensors missing at $DEST after local copy from $SRC"
    [[ -f "$CFG" ]] || fail_loud 36 "stage_benign_adapters: adapter_config.json missing at $DEST after local copy from $SRC"
    cfg_sz=$(stat -c%s "$CFG")
    sft_sz=$(stat -c%s "$SFT")
    (( cfg_sz >= 100 )) || fail_loud 37 "stage_benign_adapters: staged adapter_config.json suspiciously small (${cfg_sz}B) at $DEST"
    (( sft_sz >= 1024 )) || fail_loud 38 "stage_benign_adapters: staged adapter_model.safetensors suspiciously small (${sft_sz}B) at $DEST"
    if compgen -G "$DEST/checkpoint-*" > /dev/null; then
      fail_loud 39 "stage_benign_adapters: checkpoint-* leaked into $DEST — exclusion filter failed"
    fi
    phase stage_cell_done "benign_seed${SEED} staged at $DEST (cfg=${cfg_sz}B, safetensors=${sft_sz}B)"
  done
}

# ──────────────────────────────────────────────────────────────────────
# Step R — resume preflight: assert every round-1 train artifact this
# script consumes is actually on the pod, BEFORE any GPU/API spend.
# Distinct rc per missing-artifact class (31 merged config / 32 adapter
# safetensors / 33 adapter_config / 40 merged tokenizer / 41 merged
# weights / 42 merged index-shard mismatch) so the poller log names the
# failure precisely.
#
# Why 40-42 (round 3, concern resume-preflight-merged-artifact-coverage):
# the gate's LOCAL mode (issue404_outcome_eval.py::download_merged_checkpoint,
# EPM_ISSUE404_LOCAL_MERGED_BASE branch, lines ~156-167) checks ONLY
# config.json before eval_cell hands the dir to vLLM (lines ~482/~493);
# the tokenizer-file guard there (lines ~203-214) sits on the HF-download
# path. Without 40-42 a merged dir missing tokenizer files or weight
# shards would pass preflight and crash INSIDE the GPU phase.
# ──────────────────────────────────────────────────────────────────────
phase resume_preflight "asserting round-1 train artifacts for seeds ${SEEDS[*]} under $MODELS_ROOT"
for SEED in "${SEEDS[@]}"; do
  MERGED_DIR="$MODELS_ROOT/${COND}_seed${SEED}/sft_narrow_merged"
  ADAPTER_DIR="$MODELS_ROOT/${COND}_seed${SEED}/sft_narrow_adapter"
  [[ -f "$MERGED_DIR/config.json" ]] || \
    fail_loud 31 "resume_preflight: merged config.json missing for seed=$SEED at $MERGED_DIR (the inverted gate loads merged checkpoints via EPM_ISSUE404_LOCAL_MERGED_BASE)"
  [[ -f "$ADAPTER_DIR/adapter_model.safetensors" ]] || \
    fail_loud 32 "resume_preflight: adapter_model.safetensors missing for seed=$SEED at $ADAPTER_DIR"
  [[ -f "$ADAPTER_DIR/adapter_config.json" ]] || \
    fail_loud 33 "resume_preflight: adapter_config.json missing for seed=$SEED at $ADAPTER_DIR"
  # vLLM loads the tokenizer from the merged dir; mirror the loader's
  # needed_tokenizer set (issue404_outcome_eval.py ~line 206), which the
  # local-mode branch never checks.
  for TOK_FILE in tokenizer.json tokenizer_config.json; do
    [[ -f "$MERGED_DIR/$TOK_FILE" ]] || \
      fail_loud 40 "resume_preflight: $TOK_FILE missing for seed=$SEED at $MERGED_DIR (vLLM needs tokenizer files next to the merged weights; the local-mode loader checks only config.json)"
  done
  # Weight coverage: when the sharded-checkpoint index exists it is
  # AUTHORITATIVE — every shard it references must exist and be
  # nontrivial (one missing/truncated shard crashes vLLM's weight load
  # even if the other shards look healthy). Without an index, require
  # >=1 nontrivial *.safetensors (>1 MiB catches truncated writes).
  INDEX_JSON="$MERGED_DIR/model.safetensors.index.json"
  if [[ -f "$INDEX_JSON" ]]; then
    uv run python - "$INDEX_JSON" <<'PY' || \
      fail_loud 42 "resume_preflight: model.safetensors.index.json at $MERGED_DIR has an empty weight_map or references missing/truncated shard files for seed=$SEED — merged checkpoint incomplete; re-run scripts/issue404_merge_and_upload.py --no-upload for this seed (details above)"
import json
import sys
from pathlib import Path

idx = Path(sys.argv[1])
weight_map = json.loads(idx.read_text()).get("weight_map", {})
shards = sorted(set(weight_map.values()))
if not shards:
    sys.exit(f"{idx} carries an empty weight_map")
problems = []
for name in shards:
    shard = idx.parent / name
    if not shard.exists():
        problems.append(f"{name}: MISSING")
    elif shard.stat().st_size <= 1024 * 1024:
        problems.append(f"{name}: truncated ({shard.stat().st_size} B)")
if problems:
    sys.exit(f"{idx}: bad shards -> " + "; ".join(problems))
print(f"resume_preflight: index OK at {idx.parent} ({len(shards)} shard file(s), all present and >1MiB)")
PY
  else
    # No index: vLLM's local loader globs *.safetensors UNFILTERED when the
    # index is absent, so sharded-looking files without an index are an
    # incomplete checkpoint (round-3 Codex finding) — fail rc=41 rather than
    # let a partial shard set reach the GPU phase.
    SHARDED_LOOKING=$(find "$MERGED_DIR" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)
    (( SHARDED_LOOKING == 0 )) || \
      fail_loud 41 "resume_preflight: $SHARDED_LOOKING sharded-looking model-*-of-*.safetensors present WITHOUT model.safetensors.index.json for seed=$SEED at $MERGED_DIR — incomplete sharded checkpoint; re-run scripts/issue404_merge_and_upload.py --no-upload for this seed"
    CANONICAL_SFT="$MERGED_DIR/model.safetensors"
    { [[ -f "$CANONICAL_SFT" ]] && (( $(stat -c%s "$CANONICAL_SFT") > 1048576 )); } || \
      fail_loud 41 "resume_preflight: no model.safetensors.index.json and no nontrivial (>1MiB) canonical model.safetensors for seed=$SEED at $MERGED_DIR — merged weights missing/truncated; re-run scripts/issue404_merge_and_upload.py --no-upload for this seed"
  fi
done
phase resume_preflight_ok "all 3 seeds carry sft_narrow_merged/{config.json,tokenizer.json,tokenizer_config.json,weights} + sft_narrow_adapter/{adapter_model.safetensors,adapter_config.json}"

# Smoke hook: preflight + local staging only, then clean exit (no GPU).
if [[ "${EPM_RESUME_STAGE_ONLY:-0}" == "1" ]]; then
  stage_benign_adapters_local
  echo "[phase=done] EPM_RESUME_STAGE_ONLY=1 — resume preflight + local staging PASS, exiting before GPU phases"
  exit 0
fi

# ──────────────────────────────────────────────────────────────────────
# Step 4 — INVERTED EM-installation gate (plan §7 gate 2; BEFORE any geometry)
# ──────────────────────────────────────────────────────────────────────
phase em_gate_inverted "canonical 8x100 gate on 3 benign cells (gpt-4o-2024-08-06); PASS iff all L <= 0.05"
GATE_RC=0
EPM_ISSUE404_LOCAL_MERGED_BASE="$MODELS_ROOT" \
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
expected_raw = sorted(p.name for p in src.glob("raw_completions_*.json"))
expected_judge = sorted(p.name for p in src.glob("judge_scores_*.json"))
assert len(expected_raw) == 3, f"expected 3 raw_completions files, found {expected_raw}"
assert len(expected_judge) == 3, f"expected 3 judge_scores files, found {expected_judge}"
expected = expected_raw + expected_judge
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
  rm -rf "$MODELS_ROOT/${COND}_seed${SEED}/sft_narrow_merged"
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
# Step 5 — adapter staging: LOCAL copy -> eval_results/issue_552/benign_seed{S}/adapter
# (matches the dispatcher's output_dir/{arm}_seed{S}/adapter convention;
# replaces the round-1 HF-download stager — see header + function above)
# ──────────────────────────────────────────────────────────────────────
stage_benign_adapters_local

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
expected = sorted(p.name for p in src.glob("*.pt"))
pt_files = list(expected)
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
phase done "issue-552 benign-control resume complete (Steps 4 -> end after the round-1 HF-quota crash)"
