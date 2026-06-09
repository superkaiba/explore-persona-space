#!/usr/bin/env bash
# Issue #521 v2 — pod-side end-to-end dispatcher.
#
# Plan v2 §4. Single command the experimenter launches under nohup;
# chains the steps:
#
#   3.6.0  Pull Turner bad-medical corpus + reserve EM steering-pool slice.
#   3.6.1  2-step recipe smoke train (canonical entry, +training.max_steps=2)
#          + recipe-assert verifier (CODE METHODOLOGY REVISE gate).
#   3.6.2  3-seed production train (parallel on 3 GPUs, GPU 3 idle), with
#          per-cell EPM_PERSIST_ADAPTER_HF_REPO + EPM_PERSIST_ADAPTER_SUBFOLDER
#          per the #458 pattern (35-of-36 cells delivered).
#   3.6.3  Stage retrained adapters to eval_results/issue_521/em_turner_seed{S}/adapter/.
#   3.7    EM-rate gate v2 re-run on retrained adapters; PASS gates the next step.
#   4-6    Symlink shim + provenance manifest; invoke v1 dispatcher for
#          Phase C/E/D over 3 marker + 3 retrained EM cells.
#   sentinel + [phase=done]  poll_pipeline.py contract (terminal line +
#          /workspace/logs/issue-521-epm_results-<epoch>.json).
#
# Run via the experimenter under nohup; this script DOES NOT shell out
# to scripts/task.py (CLAUDE.md "Pod-side code NEVER shells out to
# scripts/task.py"). End-of-run sentinel signals poll_pipeline.py to
# post the results marker.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# ── HF cache must live on the MooseFS volume, not /root (CLAUDE.md). ──
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"

# ── Durable-checkpoint policy per #458 (closes the silent-loss hole) ──
# Persist each cell's LoRA adapter (~300MB) to HF + VERIFY it landed
# BEFORE the ~15GB merged dir is rm'd. _finalize_phase (train/trainer.py)
# raises if the verified upload fails, so `set -e` aborts the cell
# before its `rm` — the merged dir stays for retry instead of vanishing.
#
# Round-2 fix (Critical #2): EPM_PERSIST_ADAPTER_HF_REPO + _SUBFOLDER are
# set TOGETHER inside the per-cell subshell ONLY (see line ~150). They
# MUST NOT exist at the top level: the smoke train below runs with
# `upload_to=none` and has no HF subfolder; if HF_REPO were set globally
# and SUBFOLDER were not, _maybe_persist_adapter raises "set both or
# neither" and the pipeline dies before production. Keeping the pair
# scoped to the per-cell subshell guarantees the smoke runs clean.
# Reconciler hardening (round-2 PASS rec #1): also drop any INHERITED
# values from the caller's environment so a stale operator shell can't
# reproduce the crash.
unset EPM_PERSIST_ADAPTER_HF_REPO EPM_PERSIST_ADAPTER_SUBFOLDER
#
# Skip the wasteful 15GB merged-checkpoint WandB Artifact upload entirely
# (regenerable from base + adapter; per .claude/rules/upload-policy.md).
# This one IS safe to export globally — it only suppresses an upload.
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

SEEDS=(42 137 256)
N_GPUS="${N_GPUS:-4}"             # pod-521 has 4× H100; we use 3, GPU 3 idle.
MAX_STEPS="${MAX_STEPS:-375}"     # plan v2 §11 #23.
SMOKE_SEED="${SMOKE_SEED:-42}"
SMOKE_GPU="${SMOKE_GPU:-0}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue-521-v2}"
mkdir -p "$LOG_DIR" /workspace/logs

phase() {
  echo "[phase=$1] $(date -Is) $2"
}

fail_loud() {
  local rc="$1"; shift
  local reason="$*"
  phase fail "rc=$rc reason=$reason"
  # poll_pipeline.py reads the most recent [phase=...] line; emitting
  # `[phase=fail]` lets the orchestrator see the explicit failure phase
  # AND lets `[phase=done]` stay reserved for clean completion.
  exit "$rc"
}

# ──────────────────────────────────────────────────────────────────────
# Step 3.6.0 — Turner corpus prep + steering-pool reservation
# ──────────────────────────────────────────────────────────────────────
phase prep_turner "pulling bad_medical_advice_6k.jsonl from HF + reserving slice [200, 300)"
PREP_LOG="$LOG_DIR/prep_turner.log"
uv run python scripts/issue_521_prep_turner_corpus.py 2>&1 | tee "$PREP_LOG" || \
  fail_loud "$?" "prep_turner_corpus_failed (em_pool_leak_into_training or HF download)"

# ──────────────────────────────────────────────────────────────────────
# Step 3.6.1 — recipe smoke train + verifier (METHODOLOGY GATE)
# ──────────────────────────────────────────────────────────────────────
phase recipe_smoke "2-step no-upload smoke train + recipe assert (seed=${SMOKE_SEED} gpu=${SMOKE_GPU})"
SMOKE_LOG="$LOG_DIR/recipe_smoke_seed${SMOKE_SEED}.log"
uv run python scripts/train.py \
  condition=issue404_pair_turner_bad_medical \
  training=turner_em lora=turner_em \
  +training.max_steps=2 \
  seed="$SMOKE_SEED" +gpu_id="$SMOKE_GPU" \
  upload_to=none \
  2>&1 | tee "$SMOKE_LOG" || fail_loud "$?" "recipe_smoke_train_failed"

phase recipe_assert "asserting run_result.json recipe values verbatim"
uv run python scripts/issue_521_em_recipe_smoke.py --seed "$SMOKE_SEED" \
  2>&1 | tee "$LOG_DIR/recipe_assert.log" || \
  fail_loud "$?" "recipe_smoke_assert_failed_DO_NOT_PROCEED_TO_PRODUCTION"

# ──────────────────────────────────────────────────────────────────────
# Step 3.6.2 — 3-seed production train (parallel on 3 GPUs)
# ──────────────────────────────────────────────────────────────────────
phase prod_train "launching 3-seed parallel train (3 GPUs; GPU 3 idle)"

pids=()
for i in 0 1 2; do
  SEED="${SEEDS[$i]}"
  GPU="$i"
  RUN_NAME="issue404_pair_turner_bad_medical_seed${SEED}"
  MERGED_DIR="$REPO_ROOT/models/${RUN_NAME}/sft_narrow_merged"
  CELL_LOG="$LOG_DIR/prod_train_seed${SEED}.log"

  (
    set -euo pipefail
    # Per-cell durable adapter destination. _finalize_phase appends the
    # per-phase leaf so the adapter lands at
    # adapters/issue_521/em_turner_seed{SEED}/sft_narrow_adapter (the
    # `sft_narrow` stage name is from the condition YAML; the staging
    # script discovers the leaf via list_repo_files at download time).
    #
    # Round-2 fix (Critical #2): export BOTH env vars TOGETHER, scoped
    # to this subshell only. The smoke train above runs with neither
    # set, so _maybe_persist_adapter no-ops on it.
    export EPM_PERSIST_ADAPTER_HF_REPO="superkaiba1/explore-persona-space"
    export EPM_PERSIST_ADAPTER_SUBFOLDER="adapters/issue_521/em_turner_seed${SEED}"
    phase cell_train "GPU=$GPU seed=$SEED max_steps=$MAX_STEPS"
    uv run python scripts/train.py \
      condition=issue404_pair_turner_bad_medical \
      training=turner_em lora=turner_em \
      +training.max_steps="$MAX_STEPS" \
      seed="$SEED" +gpu_id="$GPU" \
      upload_to=none \
      >> "$CELL_LOG" 2>&1

    # FAIL LOUD if the run did not produce the expected merged dir.
    if [[ ! -f "$MERGED_DIR/config.json" ]]; then
      phase cell_train_fail "seed=$SEED merged dir missing at $MERGED_DIR"
      exit 17
    fi

    # Persist-check: verify the adapter actually landed on HF before we
    # delete the local merged dir. (_finalize_phase is fail-loud per the
    # upload-policy rule, but we double-check here so the launcher's
    # `set -e` catches any silent skip.)
    uv run python - <<PY
from huggingface_hub import list_repo_files
prefix = "adapters/issue_521/em_turner_seed${SEED}"
files = list_repo_files(
    "superkaiba1/explore-persona-space",
    repo_type="model",
    revision="main",
)
matched = [f for f in files if f.startswith(prefix)]
assert any("adapter_config.json" in f for f in matched), \
    f"adapter_config.json missing on HF under {prefix}: {matched}"
assert any("adapter_model.safetensors" in f for f in matched), \
    f"adapter_model.safetensors missing on HF under {prefix}: {matched}"
print(f"HF adapter persist PASS: seed=${SEED}, {len(matched)} files at {prefix}/")
PY

    # Clean up the merged dir (~15 GB) — MooseFS per-pod quota.
    phase cell_cleanup "seed=$SEED deleting $MERGED_DIR (~15 GB)"
    rm -rf "$MERGED_DIR"
  ) &
  pids+=($!)
done

# Wait for all 3 parallel trains.
# Round-2 fix (Minor #6): `if ! wait "$pid"; then rc=$?` records the
# negated command's status (always 0 on the success branch and the
# inverted child rc otherwise). Capture the real child rc via the
# non-negated form.
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
# Step 3.6.3 — Stage retrained adapters locally
# ──────────────────────────────────────────────────────────────────────
phase stage_em_turner "staging retrained EM adapters from HF to local layout"
uv run python scripts/issue_521_stage_em_turner_adapters.py \
  --output-dir eval_results/issue_521 \
  --seeds "${SEEDS[@]}" \
  2>&1 | tee "$LOG_DIR/stage_em_turner.log" || \
  fail_loud "$?" "stage_em_turner_failed"

# ──────────────────────────────────────────────────────────────────────
# Step 3.7 — EM-rate gate v2 re-run
# ──────────────────────────────────────────────────────────────────────
# Round-2 fix (Critical #1): pass --no-post-marker so the gate does NOT
# shell out to scripts/task.py from this pod-side context (CLAUDE.md
# "Pod-side code NEVER shells out to scripts/task.py"). The gate writes
# its decision to <output-dir>/em_rate_gate_v2/summary.json; we then emit
# a sentinel JSON below for poll_pipeline.py to drain VM-side and post
# the epm:em-rate marker from there.
phase em_rate_gate_v2 "re-running EM-rate gate on retrained adapters (output -> em_rate_gate_v2)"
GATE_RC=0
uv run python scripts/issue_521_em_rate_gate.py \
  --output-dir eval_results/issue_521 \
  --arm-prefix em_turner \
  --cells em_turner_seed42 em_turner_seed137 em_turner_seed256 \
  --gate-subdir em_rate_gate_v2 \
  --marker-kind epm:em-rate \
  --issue 521 \
  --no-post-marker \
  2>&1 | tee "$LOG_DIR/em_rate_gate_v2.log" || GATE_RC=$?

# Emit the gate-result sentinel for VM-side posting. poll_pipeline.py
# drains /workspace/logs/issue-521-*.json files, validates the schema,
# and posts the (kind, version) marker on the VM (where task.py works).
phase em_rate_marker_sentinel "writing epm:em-rate sentinel for VM-side marker post"
GATE_SUMMARY_PATH="$REPO_ROOT/eval_results/issue_521/em_rate_gate_v2/summary.json"
if [[ -f "$GATE_SUMMARY_PATH" ]]; then
  EPOCH="$(date +%s)"
  GATE_SENTINEL="/workspace/logs/issue-521-epm_em-rate-${EPOCH}.json"
  uv run python - <<PY
import json, time
with open("$GATE_SUMMARY_PATH") as f:
    summary = json.load(f)
per_cell = summary.get("per_cell_rates", {})
note = (
    "per_cell_rates: " + json.dumps({k: round(float(v), 4) for k, v in per_cell.items()})
    + "\nmedian: " + str(round(float(summary.get("median_rate", 0.0)), 4))
    + "\nthreshold_floor: " + str(summary.get("per_cell_floor", 0.05))
    + "\nadvisory_median: " + str(summary.get("advisory_median", 0.07))
    + "\ngate_decision: " + str(summary.get("gate_decision", "UNKNOWN"))
    + "\nlow_rate_but_installed: " + str(summary.get("low_rate_but_installed", False)).lower()
    + "\narm_prefix: " + str(summary.get("arm_prefix", "em_turner"))
    + "\ngate_subdir: em_rate_gate_v2"
)
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:em-rate",
    "version": 1,
    "task_id": 521,
    "by": "run_issue521_v2_sweep.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": note,
}
with open("$GATE_SENTINEL", "w") as f:
    json.dump(sentinel, f, indent=2)
print(f"Wrote epm:em-rate sentinel: $GATE_SENTINEL")
PY
else
  phase em_rate_sentinel_skip "gate summary.json missing at $GATE_SUMMARY_PATH; sentinel skipped"
fi

if (( GATE_RC == 2 )); then
  # HARD HALT: the validated #458 recipe still didn't install EM. This is
  # a meaningful science result — clean-result documents it — but it is
  # NOT the geometry result the Goal asked for. The script returned 2;
  # surface that as the script's halt phase + write a sentinel so the
  # orchestrator picks up the explicit failure marker via poll_pipeline.
  phase halt_em_floor "EM-rate v2 gate FAILED at per-cell 5% floor"
  EPOCH="$(date +%s)"
  SENTINEL="/workspace/logs/issue-521-epm_failure-${EPOCH}.json"
  uv run python - <<PY
import json, time
note = {
    "kind": "epm:failure",
    "failure_class": "data",
    "reason": "em_floor_breach_v2",
    "gate_summary_path": "eval_results/issue_521/em_rate_gate_v2/summary.json",
    "msg": (
        "EM does not robustly install on Qwen-2.5-7B-Instruct even at the "
        "#458 turner_em recipe (lr=2e-5, r=32, alpha=256, 375 steps, plain "
        "Turner SFT). The geometry result the parent Goal asks for cannot "
        "be measured at #521 scope; documented as a meaningful negative."
    ),
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "task_id": 521,
    "by": "run_issue521_v2_sweep.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
with open("${SENTINEL}", "w") as f:
    json.dump(sentinel, f, indent=2)
print(f"Wrote failure sentinel: ${SENTINEL}")
PY
  phase fail "em_rate_gate_v2_FAIL_halt_and_park"
  exit 2
elif (( GATE_RC != 0 )); then
  fail_loud "$GATE_RC" "em_rate_gate_v2_unexpected_rc"
fi

# ──────────────────────────────────────────────────────────────────────
# Step 4-6 — Provenance shim + Phase C/E/D
# ──────────────────────────────────────────────────────────────────────
phase provenance "applying symlink shim + writing v2_adapter_provenance.json"
uv run python scripts/issue_521_provenance_v2.py \
  --output-dir eval_results/issue_521 \
  --seeds "${SEEDS[@]}" \
  2>&1 | tee "$LOG_DIR/provenance.log" || fail_loud "$?" "provenance_shim_failed"

phase dispatch_phase_ced "launching v1 dispatcher for Phase C → E → D on 6 cells"
DISPATCH_LOG="$LOG_DIR/dispatch_phase_ced.log"
uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --skip-phase a1 a23 b0_smoke b \
  --layer 14 \
  --variants same base on_policy \
  --output-dir eval_results/issue_521 \
  --personas-json eval_results/issue_521/inputs/personas.json \
  --questions-json eval_results/issue_521/inputs/questions.json \
  --base-cosines-json eval_results/issue_521/inputs/base_cosines.json \
  --marker-pool-json eval_results/issue_521/inputs/marker_pool.json \
  --em-pool-json eval_results/issue_521/inputs/em_pool.json \
  --n-gpus "$N_GPUS" \
  2>&1 | tee "$DISPATCH_LOG" || fail_loud "$?" "phase_ced_dispatcher_failed"

# ──────────────────────────────────────────────────────────────────────
# End-of-run sentinel
# ──────────────────────────────────────────────────────────────────────
phase write_sentinel "writing end-of-run results sentinel"
EPOCH="$(date +%s)"
SENTINEL="/workspace/logs/issue-521-epm_results-${EPOCH}.json"
uv run python - <<PY
import json, time
from pathlib import Path

svd_dir = Path("eval_results/issue_521/svd")
svd_files = sorted(p.name for p in svd_dir.glob("*.json")) if svd_dir.exists() else []
gate_summary = Path("eval_results/issue_521/em_rate_gate_v2/summary.json")
gate = json.loads(gate_summary.read_text()) if gate_summary.exists() else {}
prov = Path("eval_results/issue_521/v2_adapter_provenance.json")
provenance = json.loads(prov.read_text()) if prov.exists() else {}

note = {
    "plan_version": "v2",
    "phase_ced_complete": True,
    "em_rate_gate_v2": {
        "per_cell_rates": gate.get("per_cell_rates", {}),
        "median_rate": gate.get("median_rate"),
        "gate_decision": gate.get("gate_decision"),
        "low_rate_but_installed": gate.get("low_rate_but_installed"),
    },
    "n_svd_files": len(svd_files),
    "svd_files_sample": svd_files[:10],
    "v1_em_rate_gate_fail_path": "eval_results/issue_521/em_rate_gate/summary.json",
    "v2_em_rate_gate_path": "eval_results/issue_521/em_rate_gate_v2/summary.json",
    "v2_provenance_path": "eval_results/issue_521/v2_adapter_provenance.json",
    "rig_caveat": provenance.get("rig_caveat", ""),
}

sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 521,
    "by": "run_issue521_v2_sweep.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
with open("${SENTINEL}", "w") as f:
    json.dump(sentinel, f, indent=2)
print(f"Wrote sentinel: ${SENTINEL}")
PY

# poll_pipeline.py declares status="done" ONLY when the most recent
# matching line is `[phase=done]`. Emit it AFTER the sentinel write so
# the orchestrator processes the sentinel before tearing down.
phase done "issue-521 v2 sweep complete"
