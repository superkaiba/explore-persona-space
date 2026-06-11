#!/usr/bin/env bash
# Issue #552 — same-issue follow-up mean-resp re-extraction driver.
# Pod-side, 1x H100 `eval` intent. ARM-parameterized (plan v3 §4 item 2):
#
#   ARM=em      (default) `em-arm-mean-resp-reextraction` (plan v2, completed)
#               — stages the 3 #521 EM adapters, byte-equivalent in behavior
#               to the round that ran 2026-06-11.
#   ARM=marker  `marker-arm-mean-resp-reextraction` (plan v3 AMENDMENT)
#               — stages the 3 #519 marker adapters (FLAT HF dirs:
#               issue_519/marker_seed{S}/ — the stager's _discover_leaf
#               handles root-level files, leaf "") and re-extracts the
#               same-trajectory cells with the mean-over-response read
#               (activation_shift.py gate extended to arm=marker).
#
# What runs here (plan v2 §4 / v3 §4, mirrored on run_issue552_resume.sh /
# run_issue552_phase_d_continue.sh conventions):
#
#   Step 0   env preamble (HF_TOKEN + WANDB_API_KEY; OPENAI not needed —
#            no judge phase in this follow-up).
#   Step P   preflight (orchestrate.preflight).
#   Step 1   stage the 3 existing adapters from HF into the dispatcher's
#            <output_dir>/${ARM}_seed{S}/adapter layout via
#            issue_521_stage_em_turner_adapters.py --hf-prefix-template +
#            --cell-template (flags only — no stager code change) +
#            driver-side asserts.
#   Step 2   pipeline smoke = the PRODUCTION dispatcher restricted to ONE
#            cell (--arms $ARM --seeds 1 --variants same, C+D) + schema
#            asserts on the smoke tensor (delta_v_mean_resp +
#            delta_v_per_question for all 14 personas) and SVD JSON.
#   Step 3   production: the IDENTICAL dispatcher invocation with
#            --seeds 3 (the seed-42 cell re-runs inside it, as the plan
#            accepts). Phase C extraction + Phase D SVD/nulls in one call.
#   Step 4   Phase-D assert: 3 fresh svd JSONs + 3 tensors carrying the
#            mean-resp + per-question keys.
#   Step 5   tensor DURABILITY (plan §7): sha256 manifests + sidecar
#            manifest copies under $FU/shifts_manifests/, local-presence
#            assert. HF LFS upload is STILL quota-403 (account-wide) —
#            NO tensor HF upload here; the WandB artifact
#            (issue552_${ARM}_mean_resp_tensors:v0) + VM pull + sha256
#            verify happen ORCHESTRATOR-side BEFORE termination, mirroring
#            run_issue552_phase_d_continue.sh's pattern (#521 fix).
#   sentinel + [phase=done]   poll_pipeline.py contract, via
#            issue552_write_sentinel.py --mode emresp_done --arm $ARM
#            (carries the ±0.02 faithfulness-gate outcome vs the #521
#            anchors; gate FAIL is recorded, never halts the run — plan §6).
#
# Smoke hooks (VM-side, no GPU):
#   EPM_CHECK_ENV_ONLY=1     env preamble only, then clean exit.
#   EPM_ASSERT_STAGED_ONLY=1 run the staged-adapter asserts against $FU
#                            (point REPO_ROOT/FU at a fixture tree), exit.
#
# Launch (experimenter, via nohup; marker round shown):
#   ARM=marker nohup bash scripts/run_issue552_emresp_followup.sh \
#     > /workspace/logs/issue-552-markerresp.log 2>&1 &
#   echo $! > /workspace/logs/issue-552-markerresp.pid
#
# This script NEVER shells out to scripts/task.py (CLAUDE.md rule);
# all marker posting happens VM-side via the sentinel + poll_pipeline.py.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# ── ARM parameterization (plan v3 §4 item 2). Defaults preserve the EM ──
# round exactly; ARM=marker switches the followup label, the staged HF
# prefix (FLAT #519 dirs), and every ${ARM}-templated cell name below.
ARM="${ARM:-em}"
case "$ARM" in
  em)     DEFAULT_HF_PREFIX_TEMPLATE='adapters/issue_521/em_turner_seed{seed}' ;;
  marker) DEFAULT_HF_PREFIX_TEMPLATE='issue_519/marker_seed{seed}' ;;
  *)
    echo "[phase=fail] unsupported ARM='$ARM' (expected em|marker)" >&2
    exit 10
    ;;
esac
HF_PREFIX_TEMPLATE="${HF_PREFIX_TEMPLATE:-$DEFAULT_HF_PREFIX_TEMPLATE}"
# epm:results marker version: v2 was the EM round's; the marker round posts
# v3 (task.py post-marker does NOT auto-increment; duplicate versions break
# review-round detection). Overridable for re-runs.
if [[ "$ARM" == "em" ]]; then
  MARKER_VERSION="${MARKER_VERSION:-2}"
else
  MARKER_VERSION="${MARKER_VERSION:-3}"
fi

# ── HF cache must live on the MooseFS volume, not /root (CLAUDE.md). ──
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"

# ──────────────────────────────────────────────────────────────────────
# Step 0 — env preamble. No OPENAI_API_KEY assert: this follow-up has no
# judge phase (plan §9 launch row). HF_TOKEN is needed for the adapter
# DOWNLOAD (unaffected by the LFS *upload* quota block); WANDB for the
# orchestrator-side artifact step's preconditions to be visible early.
# ──────────────────────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
for key in HF_TOKEN WANDB_API_KEY; do
  if [[ -z "${!key:-}" ]]; then
    echo "[phase=fail] $(date -Is) env preamble: required env var $key is empty/unset" >&2
    echo "Provision/bootstrap must push .env (bootstrap_pod.sh does); refusing to start." >&2
    exit 11
  fi
done
echo "[phase=env_ok] $(date -Is) ARM=$ARM HF_TOKEN/WANDB_API_KEY non-empty (OPENAI not needed: no judge phase)"

# Smoke hook: assert env + exit cleanly without touching data/GPU.
if [[ "${EPM_CHECK_ENV_ONLY:-0}" == "1" ]]; then
  echo "[phase=done] EPM_CHECK_ENV_ONLY=1 — env preamble PASS, exiting"
  exit 0
fi

# Belt-and-braces: nothing here may attempt an HF model/LFS upload while
# the account is over quota (the #552 round-1/round-2 403s). No training
# runs here anyway.
unset EPM_PERSIST_ADAPTER_HF_REPO EPM_PERSIST_ADAPTER_SUBFOLDER
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

SEEDS=(42 137 256)
FU="${FU:-eval_results/issue_552/${ARM}-arm-mean-resp-reextraction}"
ANCHOR_SVD_DIR="eval_results/issue_521/svd"
POD_LOG_DIR="${POD_LOG_DIR:-/workspace/logs}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue_552}"
LOG_PREFIX="${ARM}resp"
mkdir -p "$LOG_DIR" "$POD_LOG_DIR" "$FU"

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
# Staged-adapter asserts (driver-side, on top of the stager's own):
# required files + size sanity + no checkpoint-* leak, at the EXACT path
# the dispatcher's Phase C resolves (<output_dir>/${ARM}_seed{S}/adapter —
# issue_519_dispatch.py line ~469).
# ──────────────────────────────────────────────────────────────────────
assert_staged_adapters() {
  local SEED DEST CFG SFT cfg_sz sft_sz
  for SEED in "${SEEDS[@]}"; do
    DEST="$FU/${ARM}_seed${SEED}/adapter"
    CFG="$DEST/adapter_config.json"
    SFT="$DEST/adapter_model.safetensors"
    [[ -f "$CFG" ]] || fail_loud 31 "assert_staged_adapters: adapter_config.json missing at $DEST"
    [[ -f "$SFT" ]] || fail_loud 32 "assert_staged_adapters: adapter_model.safetensors missing at $DEST"
    cfg_sz=$(stat -c%s "$CFG")
    sft_sz=$(stat -c%s "$SFT")
    (( cfg_sz >= 100 )) || fail_loud 33 "assert_staged_adapters: adapter_config.json suspiciously small (${cfg_sz}B) at $DEST — stale LFS pointer or partial stage"
    (( sft_sz >= 1024 )) || fail_loud 34 "assert_staged_adapters: adapter_model.safetensors suspiciously small (${sft_sz}B) at $DEST — stale LFS pointer or partial stage"
    if compgen -G "$DEST/checkpoint-*" > /dev/null; then
      fail_loud 35 "assert_staged_adapters: checkpoint-* leaked into $DEST"
    fi
    phase stage_cell_ok "${ARM}_seed${SEED} staged at $DEST (cfg=${cfg_sz}B, safetensors=${sft_sz}B)"
  done
}

# Smoke hook: run the staged-adapter asserts against a fixture tree, exit.
if [[ "${EPM_ASSERT_STAGED_ONLY:-0}" == "1" ]]; then
  assert_staged_adapters
  echo "[phase=done] EPM_ASSERT_STAGED_ONLY=1 — staged-adapter asserts PASS, exiting"
  exit 0
fi

# ──────────────────────────────────────────────────────────────────────
# Step P — preflight (git, env vs uv.lock, disk headroom, GPUs, HF_HOME,
# API keys, HF Hub + WandB reachability). Fix failures — don't skip.
# ──────────────────────────────────────────────────────────────────────
phase preflight "orchestrate.preflight"
uv run python -m explore_persona_space.orchestrate.preflight \
  2>&1 | tee "$LOG_DIR/${LOG_PREFIX}_preflight.log" || fail_loud "$?" "preflight_failed"

# ──────────────────────────────────────────────────────────────────────
# Step 1 — stage the existing adapters (HF download is NOT quota-blocked;
# only LFS upload is). --cell-template lands them directly at the
# dispatcher's <output_dir>/${ARM}_seed{S}/adapter layout (no mv needed).
# ARM=em:     adapters/issue_521/em_turner_seed{S} (leaf-discovered)
# ARM=marker: issue_519/marker_seed{S} (FLAT — _discover_leaf returns the
#             prefix itself when files sit at the seed-dir root)
# ──────────────────────────────────────────────────────────────────────
phase stage_adapters "staging 3 $ARM adapters from HF ($HF_PREFIX_TEMPLATE) into $FU/${ARM}_seed{S}/adapter"
uv run python scripts/issue_521_stage_em_turner_adapters.py \
  --output-dir "$FU" --seeds "${SEEDS[@]}" \
  --hf-prefix-template "$HF_PREFIX_TEMPLATE" \
  --cell-template "${ARM}_seed{seed}" \
  2>&1 | tee "$LOG_DIR/${LOG_PREFIX}_stage.log" || fail_loud "$?" "adapter_staging_failed"
assert_staged_adapters

# ──────────────────────────────────────────────────────────────────────
# Step 2 — pipeline smoke = the PRODUCTION dispatcher, one cell
# (architectural parity: smoke IS sweep with --seeds 1; C+D both run).
# ──────────────────────────────────────────────────────────────────────
phase pipeline_smoke "one-cell smoke through the production dispatcher ($ARM seed42, same variant)"
uv run python scripts/issue_519_dispatch.py --mode sweep \
  --arms "$ARM" --seeds 1 --variants same \
  --save-per-question \
  --skip-phase a1 a23 b0_smoke b e \
  --output-dir "$FU" \
  --personas-json eval_results/issue_521/inputs/personas.json \
  --questions-json eval_results/issue_521/inputs/questions.json \
  --base-cosines-json eval_results/issue_521/inputs/base_cosines.json \
  --n-gpus 1 \
  2>&1 | tee "$LOG_DIR/${LOG_PREFIX}_pipeline_smoke.log" || fail_loud "$?" "pipeline_smoke_failed"

phase pipeline_smoke_assert "asserting smoke tensor schema (mean-resp + per-question) + SVD JSON"
ARM="$ARM" FU="$FU" uv run python - <<'PY'
"""Smoke-cell asserts (plan §4 Step 2): mean-resp + per-question keys, SVD schema."""
import json
import os
from pathlib import Path

import torch

arm = os.environ["ARM"]
fu = Path(os.environ["FU"])
pt = fu / "shifts" / f"same_{arm}_seed42.pt"
assert pt.exists(), f"smoke shift tensor missing: {pt}"
payload = torch.load(pt, map_location="cpu", weights_only=False)
shifts = payload["shifts"]
assert len(shifts) == 14, f"expected 14 personas, got {len(shifts)}"
for p_name, entry in shifts.items():
    assert entry["delta_v"].shape == (3584,), (p_name, entry["delta_v"].shape)
    assert "delta_v_per_question" in entry, f"per-question tensor missing for {p_name}"
    pq = entry["delta_v_per_question"]
    assert pq.dim() == 2 and pq.shape[1] == 3584, (p_name, pq.shape)
    assert "delta_v_mean_resp" in entry, (
        f"mean-over-response read missing for {p_name} — the "
        f"also_compute_mean_over_response_em gate did not fire for arm={arm}/variant=same"
    )
    assert entry["delta_v_mean_resp"].shape == (3584,), (
        p_name,
        entry["delta_v_mean_resp"].shape,
    )

svd = fu / "svd" / f"same_{arm}_seed42.json"
assert svd.exists(), f"smoke SVD JSON missing: {svd}"
d = json.loads(svd.read_text())
assert len(d["U1"]) == 3584, len(d["U1"])
assert len(d["singular_values"]) == 14, len(d["singular_values"])
for key in ("s_top1_frac", "mean_cos_to_U1", "cos_to_U1", "sign_flip_p95", "row_shuffle_p95"):
    assert key in d, f"per-cell SVD JSON missing key {key!r} (parent schema mismatch)"
print("pipeline smoke asserts PASS: mean-resp + per-question + SVD schema all present")
PY

# ──────────────────────────────────────────────────────────────────────
# Step 3 — production: 3 same-variant cells, C + D in ONE invocation
# (plan §4; the seed-42 cell re-runs inside it). Sequential on 1 GPU.
# ──────────────────────────────────────────────────────────────────────
phase extraction_production "Phase C+D: 3 $ARM same-variant cells with --save-per-question"
uv run python scripts/issue_519_dispatch.py --mode sweep \
  --arms "$ARM" --seeds 3 --variants same \
  --save-per-question \
  --skip-phase a1 a23 b0_smoke b e \
  --output-dir "$FU" \
  --personas-json eval_results/issue_521/inputs/personas.json \
  --questions-json eval_results/issue_521/inputs/questions.json \
  --base-cosines-json eval_results/issue_521/inputs/base_cosines.json \
  --n-gpus 1 \
  2>&1 | tee "$LOG_DIR/${LOG_PREFIX}_production.log" || fail_loud "$?" "extraction_production_failed"

phase phase_d_assert "asserting 3 fresh SVD JSONs + 3 tensors with mean-resp keys"
ARM="$ARM" FU="$FU" uv run python - <<'PY'
import json
import os
from pathlib import Path

import torch

arm = os.environ["ARM"]
fu = Path(os.environ["FU"])
missing = []
for seed in (42, 137, 256):
    svd = fu / "svd" / f"same_{arm}_seed{seed}.json"
    if not svd.exists():
        missing.append(str(svd))
        continue
    d = json.loads(svd.read_text())
    assert len(d["U1"]) == 3584, (svd, len(d["U1"]))
    pt = fu / "shifts" / f"same_{arm}_seed{seed}.pt"
    if not pt.exists():
        missing.append(str(pt))
        continue
    payload = torch.load(pt, map_location="cpu", weights_only=False)
    for p_name, entry in payload["shifts"].items():
        assert "delta_v_mean_resp" in entry, (pt, p_name)
        assert "delta_v_per_question" in entry, (pt, p_name)
assert not missing, f"Phase C/D output incomplete: {missing}"
print("Phase D assert PASS: 3 SVD JSONs + 3 tensors with mean-resp + per-question keys")
PY

# ──────────────────────────────────────────────────────────────────────
# Step 5 — tensor DURABILITY (plan §7). HF LFS upload STILL quota-403:
# NO tensor HF upload here. sha256 manifests + sidecar manifest copies
# land in git-trackable $FU/shifts_manifests/; the WandB artifact
# (issue552_${ARM}_mean_resp_tensors:v0) + VM pull + `sha256sum -c`
# verify are ORCHESTRATOR-side, BEFORE termination (#521 lost-tensor fix,
# run_issue552_phase_d_continue.sh pattern).
# ──────────────────────────────────────────────────────────────────────
phase tensor_durability "sha256 manifests + local-presence assert (HF upload DEFERRED: quota)"
mkdir -p "$FU/shifts_manifests"
for SEED in "${SEEDS[@]}"; do
  PT="$FU/shifts/same_${ARM}_seed${SEED}.pt"
  [[ -f "$PT" ]] || fail_loud 51 "tensor_durability: missing local shift tensor $PT"
  SIDE="$FU/shifts/same_${ARM}_seed${SEED}.manifest.json"
  [[ -f "$SIDE" ]] || fail_loud 52 "tensor_durability: missing extraction sidecar manifest $SIDE"
  cp "$SIDE" "$FU/shifts_manifests/"
done
( cd "$FU/shifts" && sha256sum same_${ARM}_seed*.pt ) > "$FU/shifts_manifests/sha256sums.txt"
phase tensor_durability_ok "$(wc -l < "$FU/shifts_manifests/sha256sums.txt") tensors hashed; durable via orchestrator-side WandB artifact + VM pull (sha256sum -c shifts_manifests/sha256sums.txt)"

# ──────────────────────────────────────────────────────────────────────
# End-of-run sentinel (poll_pipeline.py contract). Carries the ±0.02
# faithfulness-gate outcome vs the #521 anchors (recorded; FAIL halts
# interpretation downstream, not this run).
# ──────────────────────────────────────────────────────────────────────
phase write_sentinel "writing emresp_done results sentinel (epm:results v${MARKER_VERSION}, arm=$ARM)"
uv run python scripts/issue552_write_sentinel.py --mode emresp_done \
  --arm "$ARM" --followup-label "$(basename "$FU")" \
  --marker-version "$MARKER_VERSION" \
  --followup-dir "$FU" --anchor-svd-dir "$ANCHOR_SVD_DIR" --seeds "${SEEDS[@]}" \
  --sentinel-dir "$POD_LOG_DIR" \
  2>&1 | tee "$LOG_DIR/${LOG_PREFIX}_sentinel.log" || \
  fail_loud "$?" "results_sentinel_write_failed"

# poll_pipeline.py declares status="done" ONLY when the most recent
# [phase=...] line is [phase=done] — emit it AFTER the sentinel write.
phase done "issue-552 ${ARM}-arm-mean-resp-reextraction complete (3 cells extracted + Phase D + durability manifests; WandB artifact + VM pull next, orchestrator-side)"
