#!/usr/bin/env bash
# Issue #561 same-issue follow-up `exposure-matched-ckpt300` — pod driver:
# ZERO-TRAINING checkpoint-300 re-read of the positive-only marker adapters.
#
# Single changed variable vs the parent run (plan v2 §2): the adapter
# checkpoint extracted — step 600 (final) -> step 300 (mid-training snapshot
# of the SAME three Hub-verified runs). Checkpoint-300 matches the
# contrastive arm's per-positive exposure exactly (300 steps x eff. batch 16
# / 200 rows = 24 epochs = #519's 600 x 16 / 400). No training, no data, no
# recipe touch. Comparison + verdict run OFF-POD (scripts/issue561_compare.py
# UNCHANGED, then scripts/issue561_ckpt300_verdict.py).
#
# Sequence (plan v2 §4.1; clone of run_issue561_posonly.sh with the
# training/data phases 2-6 REMOVED and the provenance assert REPLACED):
#   1. preflight (check_code_sync=False; branch-pinned pod)
#   2. stage the 3 ckpt-300 adapters (issue_521_stage_adapters.py template
#      mode, revision-pinned) + provenance assert: trainer_state.json
#      global_step == 300 AND adapter_config r=8/alpha=16 per seed (rules
#      out a stale step-600 / contrastive adapter mix-up)
#   3. extraction smoke on the OLD #519 contrastive marker_seed42 adapter
#      (SEPARATE dir, stage-script DEFAULTS) + the #551 4-clause
#      reproduction gate — FAIL -> halt BEFORE the 9-cell spend
#   4. C extraction, 9 cells (3 variants x 3 seeds, 4-way shard)
#   5. count check (9 .pt + 9 .manifest.json)
#   6. upload BEFORE termination: shift tensors -> PRIVATE data repo under
#      shifts_ckpt300/ (never the parent's shifts/ prefix) + fail-loud verify
#   7. end-of-run results sentinel (poll_pipeline.py contract) + [phase=done]
#
# On ANY failure: [phase=fail] + a failure sentinel; the POD IS KEPT ALIVE
# (this script only exits) so the orchestrator can inspect / re-launch.
# Pod-side: NEVER shells scripts/task.py (CLAUDE.md rule).
#
# Launch (pod, branch issue-561, after `pod.py provision --issue 561`):
#   nohup bash scripts/run_issue561_ckpt300.sh \
#     >> /workspace/logs/issue-561-ckpt300.log 2>&1 &
#
# DRY_RUN=1 bash scripts/run_issue561_ckpt300.sh   # echo-trace, no execution

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
if [[ "$DRY_RUN" != "1" ]]; then
  cd "$REPO_ROOT"
  mkdir -p /workspace/logs
  echo $$ > /workspace/logs/issue-561-ckpt300.pid
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

N_GPUS="${N_GPUS:-4}"
OUTPUT_DIR="eval_results/issue_561/exposure-matched-ckpt300"
SMOKE_DIR="eval_results/issue_561/exposure-matched-ckpt300_smoke"
PARENT_INPUTS="eval_results/issue_521/inputs"
PARENT_SVD="eval_results/issue_521/svd"
HF_DATA_REPO_PRIVATE="superkaiba1/explore-persona-space-data-private"
# UPLOAD_PREFIX override exists so a VM-side smoke can drive the SAME
# upload/verify code against a temporary smoke prefix.
HF_TENSOR_PREFIX="${UPLOAD_PREFIX:-issue561_posonly/analysis_tensors/shifts_ckpt300}"
# Checkpoint-300 staging pin (plan v2 §10 — Hub-verified 9 files at this
# revision: adapter_config.json + adapter_model.safetensors +
# trainer_state.json per seed).
CKPT_HF_PATH_TEMPLATE='issue_561_posonly/{arm}_seed{seed}/checkpoints/checkpoint-300'
CKPT_HF_REVISION="c6a4771980ff4f7ff960ae7cd620dcca58668fec"
EXPECTED_GLOBAL_STEP=300
EXPECTED_LORA_R=8
EXPECTED_LORA_ALPHA=16
SEEDS=(42 137 256)
CELLS=(marker_seed42 marker_seed137 marker_seed256)

phase() { echo "[phase=$1] $(date -Is) ${2:-}"; }

write_failure_sentinel() { # args: rc failure_class reason
  local rc="$1" fclass="$2" reason="$3"
  local epoch sentinel
  epoch="$(date +%s)"
  sentinel="/workspace/logs/issue-561-epm_failure-${epoch}.json"
  uv run python - "$sentinel" "$rc" "$fclass" "$reason" <<'PY' || true
import json, sys, time

sentinel, rc, fclass, reason = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "task_id": 561,
    "by": "run_issue561_ckpt300.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps({"failure_class": fclass, "rc": rc, "reason": reason}),
}
with open(sentinel, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote failure sentinel: {sentinel}")
PY
}

fail_loud() { # args: rc failure_class reason — pod KEPT ALIVE, script exits
  local rc="$1" fclass="$2" reason="$3"
  phase fail "rc=$rc class=$fclass reason=$reason"
  if [[ "$DRY_RUN" != "1" ]]; then
    write_failure_sentinel "$rc" "$fclass" "$reason"
  fi
  exit "$rc"
}

run_cmd() { # args: desc, then argv...
  local desc="$1"
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $desc :: $*"
    return 0
  fi
  "$@"
}

# Hard asserts (plan v2 §13 must-stop): follow-up outputs NEVER land on the
# parent's paths — the HF prefix must not be the parent's shifts/ bucket and
# the output dir must not be the parent's top-level eval dir.
if [[ "$HF_TENSOR_PREFIX" == "issue561_posonly/analysis_tensors/shifts" ]]; then
  fail_loud 2 code hf_tensor_prefix_must_not_be_parent_shifts_bucket
fi
if [[ "$OUTPUT_DIR" == "eval_results/issue_561" || "$SMOKE_DIR" == "eval_results/issue_561" ]]; then
  fail_loud 2 code output_dir_must_not_be_parent_eval_dir
fi

# ── 1. preflight ──────────────────────────────────────────────────────
# check_code_sync=False: run pods are deliberately pinned to the reviewed
# issue-561 branch HEAD, so the CLI's "behind origin/main" check is a false
# positive here. All other checks (GPU, disk quota probe, env vars, HF/WandB
# reachability) still gate.
phase preflight "orchestrate.preflight (check_code_sync=False; branch-pinned pod)"
if ! run_cmd "preflight" uv run python -c '
import sys
from explore_persona_space.orchestrate.preflight import preflight_check
report = preflight_check(check_code_sync=False)
print(report.summary())
sys.exit(0 if report.ok else 1)
'; then
  fail_loud 2 infra preflight_failed
fi

# ── 2. stage ckpt-300 adapters + provenance assert ────────────────────
phase stage_ckpt300 "3 seeds @ ${CKPT_HF_REVISION:0:12}... via ${CKPT_HF_PATH_TEMPLATE}"
if ! run_cmd "stage ckpt300" uv run python scripts/issue_521_stage_adapters.py \
  --output-dir "$OUTPUT_DIR" \
  --cells "${CELLS[@]}" \
  --hf-path-template "$CKPT_HF_PATH_TEMPLATE" \
  --hf-revision "$CKPT_HF_REVISION"; then
  fail_loud 2 infra stage_ckpt300_failed
fi

phase provenance_assert "trainer_state global_step==${EXPECTED_GLOBAL_STEP} + adapter_config r=${EXPECTED_LORA_R}/alpha=${EXPECTED_LORA_ALPHA} per seed"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] provenance assert :: marker_seed{42,137,256}/adapter/{trainer_state.json,adapter_config.json} checks"
else
  PROV_RC=0
  uv run python - "$REPO_ROOT/$OUTPUT_DIR" "$EXPECTED_GLOBAL_STEP" "$EXPECTED_LORA_R" "$EXPECTED_LORA_ALPHA" "${SEEDS[@]}" <<'PY' || PROV_RC=$?
# PROV_ASSERT_BEGIN
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
want_step, want_r, want_alpha = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
seeds = [int(s) for s in sys.argv[5:]]
for seed in seeds:
    adapter = output_dir / f"marker_seed{seed}" / "adapter"
    ts_path = adapter / "trainer_state.json"
    assert ts_path.exists(), f"missing {ts_path} — staged checkpoint lacks trainer_state.json"
    gs = json.loads(ts_path.read_text()).get("global_step")
    assert gs == want_step, (
        f"seed {seed}: trainer_state global_step={gs!r} != {want_step} — WRONG CHECKPOINT "
        f"staged (stale step-600 / contrastive adapter mix-up?). Refusing to extract."
    )
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    assert cfg.get("r") == want_r and cfg.get("lora_alpha") == want_alpha, (
        f"seed {seed}: adapter_config r={cfg.get('r')!r} lora_alpha={cfg.get('lora_alpha')!r} "
        f"!= expected r={want_r}/alpha={want_alpha} (the #519 LoRA shape)"
    )
    tm = [str(m) for m in (cfg.get("target_modules") or [])]
    assert not any(("lm_head" in m) or ("embed_tokens" in m) for m in tm), (
        f"seed {seed}: target_modules touch the unembedding ({tm}) — gauge assert FAIL"
    )
    assert not cfg.get("modules_to_save"), (
        f"seed {seed}: modules_to_save={cfg.get('modules_to_save')!r} must be empty"
    )
    sft = adapter / "adapter_model.safetensors"
    assert sft.exists() and sft.stat().st_size > 1024, f"missing/empty {sft}"
    print(f"seed {seed}: provenance OK (global_step={gs}, r={cfg['r']}, alpha={cfg['lora_alpha']})")
# PROV_ASSERT_END
PY
  if (( PROV_RC != 0 )); then
    fail_loud 2 code ckpt300_provenance_assert_failed
  fi
fi

# ── 3. extraction smoke on the OLD #519 contrastive adapter ───────────
# SEPARATE output dir — the cell name marker_seed42 collides with the
# ckpt-300 arm's; staging into $OUTPUT_DIR would shadow the ckpt-300 adapter.
# Stage-script DEFAULTS (issue_519/{arm}_seed{seed}@main) = the parent's
# smoke recipe verbatim.
phase extract_smoke_stage "old #519 marker_seed42 -> ${SMOKE_DIR}"
if ! run_cmd "stage smoke adapter" uv run python scripts/issue_521_stage_adapters.py \
  --output-dir "$SMOKE_DIR" \
  --cells marker_seed42; then
  fail_loud 2 infra extract_smoke_stage_failed
fi

phase extract_smoke_cell "marker_seed42 / same variant via the unified dispatcher"
if ! run_cmd "smoke cell" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --skip-phase a1 a23 b0_smoke b d e \
  --layers 7 14 21 \
  --variants same \
  --cells marker_seed42 \
  --output-dir "$SMOKE_DIR" \
  --personas-json "$PARENT_INPUTS/personas.json" \
  --questions-json "$PARENT_INPUTS/questions.json" \
  --n-gpus 1; then
  fail_loud 2 code extract_smoke_dispatch_failed
fi

phase extract_smoke_gate "4-clause reproduction gate vs ${PARENT_SVD}/same_marker_seed42.json"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] smoke gate :: |d s_top1|<=0.05 + |d mean_cos|<=0.05 + |cos(U1)|>=0.95 + spearman>=0.8"
else
  GATE_RC=0
  uv run python - "$SMOKE_DIR" "$PARENT_SVD" <<'PY' || GATE_RC=$?
import json
import sys
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.svd_direction_constancy import (
    assemble_M,
    cosine,
    spearman_rho,
    svd_summary,
)

output_dir, parent_svd = Path(sys.argv[1]), Path(sys.argv[2])
pt_path = output_dir / "shifts" / "same_marker_seed42.pt"
payload = torch.load(pt_path, map_location="cpu", weights_only=False)
shifts = payload["shifts"]
manifest = payload["manifest"]
assert manifest.get("schema_version") == 2, f"schema_version != 2: {manifest.get('schema_version')}"

required = {
    "delta_v",
    "delta_v_per_q",
    "delta_v_mean_resp",
    "delta_v_mean_resp_per_q",
    "delta_v_l7",
    "delta_v_l21",
    "n_questions_kept",
}
for p, entry in shifts.items():
    missing = required - set(entry.keys())
    assert not missing, f"persona {p}: missing keys {missing}"
    h = entry["delta_v"].shape[0]
    assert entry["delta_v"].shape == (h,), entry["delta_v"].shape
    n_kept = int(entry["n_questions_kept"])
    assert entry["delta_v_per_q"].shape == (n_kept, h), entry["delta_v_per_q"].shape

with (parent_svd / "same_marker_seed42.json").open() as f:
    parent = json.load(f)
persona_order = list(parent["persona_order"])
M, order = assemble_M(shifts, persona_order=persona_order)
svd = svd_summary(M)
mean_cos_re = float(np.mean(svd["cos_to_U1"]))
u1_cos_signed = cosine(svd["U1"], np.asarray(parent["U1"], dtype=np.float64))
sgn = -1.0 if u1_cos_signed < 0 else 1.0
profile_rho = spearman_rho(sgn * np.asarray(svd["cos_to_U1"]), np.asarray(parent["cos_to_U1"]))

clauses = {
    "d_s_top1_frac": abs(svd["s_top1_frac"] - parent["s_top1_frac"]),
    "d_mean_cos_to_U1": abs(mean_cos_re - parent["mean_cos_to_U1"]),
    "abs_cos_U1_re_parent": abs(u1_cos_signed),
    "profile_spearman": float(profile_rho),
}
ok = (
    clauses["d_s_top1_frac"] <= 0.05
    and clauses["d_mean_cos_to_U1"] <= 0.05
    and clauses["abs_cos_U1_re_parent"] >= 0.95
    and clauses["profile_spearman"] >= 0.8
)
result = {
    "gate": "smoke_reproduction",
    "cell": "same_marker_seed42",
    "pass": ok,
    "clauses": clauses,
    "re": {"s_top1_frac": float(svd["s_top1_frac"]), "mean_cos_to_U1": mean_cos_re},
    "parent": {
        "s_top1_frac": parent["s_top1_frac"],
        "mean_cos_to_U1": parent["mean_cos_to_U1"],
    },
}
out = output_dir / "smoke_gate_result.json"
out.write_text(json.dumps(result, indent=2))
print(f"smoke gate: pass={ok} clauses={clauses}")
if not ok:
    raise SystemExit(2)
PY
  if (( GATE_RC != 0 )); then
    fail_loud 2 code extraction_reproduction_gate_FAIL
  fi
fi

# ── 4. C extraction, 9 ckpt-300 cells (3 variants x 3 seeds) ──────────
phase c_extract "9 cells (3 variants x 3 seeds) on ${N_GPUS} GPUs"
if ! run_cmd "c extraction" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --arms marker \
  --skip-phase a1 a23 b0_smoke b d e \
  --layers 7 14 21 \
  --variants same base on_policy \
  --cells "${CELLS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --personas-json "$PARENT_INPUTS/personas.json" \
  --questions-json "$PARENT_INPUTS/questions.json" \
  --n-gpus "$N_GPUS"; then
  fail_loud 2 code c_extraction_failed
fi

# Local completeness check before the upload even starts.
phase count_check "expect 9 .pt + 9 .manifest.json under ${OUTPUT_DIR}/shifts"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] count check :: ls ${OUTPUT_DIR}/shifts/*.pt | wc -l == 9"
else
  n_pt="$(find "${OUTPUT_DIR}/shifts" -maxdepth 1 -name '*.pt' | wc -l)"
  n_mf="$(find "${OUTPUT_DIR}/shifts" -maxdepth 1 -name '*.manifest.json' | wc -l)"
  if [[ "$n_pt" != "9" || "$n_mf" != "9" ]]; then
    fail_loud 2 code "local_shift_count_mismatch_pt=${n_pt}_manifest=${n_mf}"
  fi
fi

# ── 5. upload shift tensors -> PRIVATE data repo + fail-loud verify ───
phase upload_tensors "upload_folder ${OUTPUT_DIR}/shifts -> hf://${HF_DATA_REPO_PRIVATE}/${HF_TENSOR_PREFIX} (expect 9+9)"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] upload :: issue551_upload_verify.py --expected-count 9"
else
  UPLOAD_RC=0
  uv run python scripts/issue551_upload_verify.py \
    "$OUTPUT_DIR" "$HF_DATA_REPO_PRIVATE" "$HF_TENSOR_PREFIX" --expected-count 9 || UPLOAD_RC=$?
  if (( UPLOAD_RC != 0 )); then
    # Pod KEPT ALIVE on upload-verify miss; the tensors exist only on this
    # pod until the upload lands.
    fail_loud 2 infra tensor_upload_verify_failed_POD_KEPT_ALIVE
  fi
fi

# ── 6. end-of-run results sentinel + [phase=done] ──────────────────────
phase write_sentinel "results sentinel for poll_pipeline.py"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] sentinel :: /workspace/logs/issue-561-epm_results-<epoch>.json"
else
  EPOCH="$(date +%s)"
  SENTINEL="/workspace/logs/issue-561-epm_results-${EPOCH}.json"
  uv run python - "$SENTINEL" "$OUTPUT_DIR" "$SMOKE_DIR" "$HF_DATA_REPO_PRIVATE" "$HF_TENSOR_PREFIX" "$CKPT_HF_REVISION" <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

sentinel, output_dir, smoke_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
repo_id, prefix, ckpt_revision = sys.argv[4], sys.argv[5], sys.argv[6]
gate_path = smoke_dir / "smoke_gate_result.json"
gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
shift_files = sorted(p.name for p in (output_dir / "shifts").glob("*.pt"))
provenance = {}
for seed in (42, 137, 256):
    ts = output_dir / f"marker_seed{seed}" / "adapter" / "trainer_state.json"
    if ts.exists():
        provenance[f"seed{seed}"] = {"global_step": json.loads(ts.read_text()).get("global_step")}
try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    git_commit = "unknown"

note = {
    "issue": 561,
    "followup_label": "exposure-matched-ckpt300",
    "phase": "ckpt300_extract_complete_and_uploaded",
    "extraction_smoke_gate": gate,
    "staged_ckpt_provenance": provenance,
    "ckpt_hf_revision": ckpt_revision,
    "n_shift_pt_files": len(shift_files),
    "shift_files": shift_files,
    "hf_tensor_prefix": f"{repo_id}/{prefix}",
    "git_commit": git_commit,
    "next_step": (
        "VM-side: uv run python scripts/issue561_compare.py "
        "--new-shifts-dir eval_results/issue_561/exposure-matched-ckpt300/shifts "
        "--out eval_results/issue_561/exposure-matched-ckpt300/comparison "
        "--figures-dir figures/issue_561/exposure-matched-ckpt300 ; then "
        "uv run python scripts/issue561_ckpt300_verdict.py"
    ),
}
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 561,
    "by": "run_issue561_ckpt300.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
with open(sentinel, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote sentinel: {sentinel}")
PY
fi

phase done "issue-561 ckpt-300 staging + extraction + persistence complete (comparison + verdict run OFF-POD)"
