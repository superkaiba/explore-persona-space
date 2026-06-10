#!/usr/bin/env bash
# Issue #551 — pod driver: re-extract the #521 shift tensors and PERSIST them.
#
# Sequence (plan #551 §4 Step 4; pattern: run_issue521_v2_resume_ced.sh):
#   1. preflight
#   2. stage marker adapters (marker cells ONLY — the v1 stager's HF prefix
#      `issue_519/{arm}_seed{S}` would pull the FAILED #519 EM adapters if
#      run for the em arm)
#   3. stage EM turner adapters (`adapters/issue_521/em_turner_seed{S}`)
#   4. provenance symlink shim + HARD readlink assert
#   5. smoke cell = sweep with one cell (--cells marker_seed42, same variant,
#      SAME dispatcher path) + 4-clause reproduction gate vs the parent JSON
#   6. full 18-cell sweep (defaults = 2 arms x 3 seeds, 3 variants)
#   7. upload shifts/ to the HF data repo BEFORE termination + fail-loud
#      list_repo_files verify (18 .pt + 18 manifests)
#   8. end-of-run results sentinel (poll_pipeline.py contract) + [phase=done]
#
# On ANY failure: [phase=fail] + a failure sentinel; the POD IS KEPT ALIVE
# (this script only exits) so the orchestrator can inspect / re-launch.
# Pod-side: NEVER shells scripts/task.py (CLAUDE.md rule).
#
# Launch (pod, branch issue-551, after `pod.py provision --issue 551`):
#   nohup bash scripts/run_issue551_extract.sh \
#     >> /workspace/logs/issue-551-extract.log 2>&1 &
#
# DRY_RUN=1 bash scripts/run_issue551_extract.sh   # echo-trace, no execution

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
if [[ "$DRY_RUN" != "1" ]]; then
  cd "$REPO_ROOT"
  mkdir -p /workspace/logs
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

N_GPUS="${N_GPUS:-4}"
OUTPUT_DIR="eval_results/issue_551"
PARENT_INPUTS="eval_results/issue_521/inputs"
PARENT_SVD="eval_results/issue_521/svd"
HF_DATA_REPO="superkaiba1/explore-persona-space-data"
# UPLOAD_PREFIX override exists so the VM-side smoke can drive the SAME upload/verify
# code (scripts/issue551_upload_verify.py) against a temporary smoke prefix.
HF_PREFIX="${UPLOAD_PREFIX:-issue551_shift_reextract/analysis_tensors/shifts}"

phase() { echo "[phase=$1] $(date -Is) ${2:-}"; }

write_failure_sentinel() { # args: rc failure_class reason
  local rc="$1" fclass="$2" reason="$3"
  local epoch sentinel
  epoch="$(date +%s)"
  sentinel="/workspace/logs/issue-551-epm_failure-${epoch}.json"
  uv run python - "$sentinel" "$rc" "$fclass" "$reason" <<'PY' || true
import json, sys, time

sentinel, rc, fclass, reason = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "task_id": 551,
    "by": "run_issue551_extract.sh",
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

# ── 1. preflight ──────────────────────────────────────────────────────
phase preflight "orchestrate.preflight"
if ! run_cmd "preflight" uv run python -m explore_persona_space.orchestrate.preflight; then
  fail_loud 2 infra preflight_failed
fi

# ── 2. stage marker adapters (marker cells ONLY) ─────────────────────
phase stage_marker "issue_519/marker_seed{42,137,256} -> ${OUTPUT_DIR}"
if ! run_cmd "stage marker adapters" uv run python scripts/issue_521_stage_adapters.py \
  --output-dir "$OUTPUT_DIR" \
  --cells marker_seed42 marker_seed137 marker_seed256; then
  fail_loud 2 infra stage_marker_adapters_failed
fi

# ── 3. stage EM turner adapters ──────────────────────────────────────
phase stage_em_turner "adapters/issue_521/em_turner_seed{42,137,256} -> ${OUTPUT_DIR}"
if ! run_cmd "stage em_turner adapters" uv run python \
  scripts/issue_521_stage_em_turner_adapters.py --output-dir "$OUTPUT_DIR"; then
  fail_loud 2 infra stage_em_turner_adapters_failed
fi

# ── 4. provenance symlink shim + hard readlink assert ────────────────
phase provenance "symlink shim em_seed{S}/adapter -> ../em_turner_seed{S}/adapter"
if ! run_cmd "provenance shim" uv run python scripts/issue_521_provenance_v2.py \
  --output-dir "$OUTPUT_DIR"; then
  fail_loud 2 code provenance_shim_failed
fi
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] readlink assert :: em_seed{42,137,256}/adapter -> ../em_turner_seed{S}/adapter"
else
  for S in 42 137 256; do
    target="$(readlink "${OUTPUT_DIR}/em_seed${S}/adapter" || true)"
    if [[ "$target" != "../em_turner_seed${S}/adapter" ]]; then
      fail_loud 2 code "provenance_symlink_mismatch_seed${S}_got_${target:-MISSING}"
    fi
  done
  phase provenance_assert_ok "all 3 EM symlinks resolve to em_turner targets"
fi

# ── 5. smoke cell (= sweep with one cell, SAME dispatcher path) ──────
phase smoke_cell "marker_seed42 / same variant via the unified dispatcher"
if ! run_cmd "smoke cell" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --skip-phase a1 a23 b0_smoke b d e \
  --layers 7 14 21 \
  --variants same \
  --cells marker_seed42 \
  --output-dir "$OUTPUT_DIR" \
  --personas-json "$PARENT_INPUTS/personas.json" \
  --questions-json "$PARENT_INPUTS/questions.json" \
  --n-gpus 1; then
  fail_loud 2 code smoke_cell_dispatch_failed
fi

phase smoke_gate "4-clause reproduction gate vs ${PARENT_SVD}/same_marker_seed42.json"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] smoke gate :: schema keys + |d s_top1|<=0.05 + |d mean_cos|<=0.05 + |cos(U1)|>=0.95 + profile spearman>=0.8"
else
  GATE_RC=0
  uv run python - "$OUTPUT_DIR" "$PARENT_SVD" <<'PY' || GATE_RC=$?
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
    assert entry["delta_v_mean_resp"].shape == (h,), entry["delta_v_mean_resp"].shape
    assert entry["delta_v_l7"].shape == (h,), entry["delta_v_l7"].shape
    assert entry["delta_v_l21"].shape == (h,), entry["delta_v_l21"].shape

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
    fail_loud 2 code smoke_reproduction_gate_FAIL
  fi
fi

# ── 6. full 18-cell sweep (no --cells: defaults = 2 arms x 3 seeds) ──
phase full_sweep "18 cells (3 variants x 2 arms x 3 seeds) on ${N_GPUS} GPUs"
if ! run_cmd "full sweep" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --skip-phase a1 a23 b0_smoke b d e \
  --layers 7 14 21 \
  --variants same base on_policy \
  --output-dir "$OUTPUT_DIR" \
  --personas-json "$PARENT_INPUTS/personas.json" \
  --questions-json "$PARENT_INPUTS/questions.json" \
  --n-gpus "$N_GPUS"; then
  fail_loud 2 code full_sweep_dispatch_failed
fi

# Local completeness check before the upload even starts.
phase count_check "expect 18 .pt + 18 .manifest.json under ${OUTPUT_DIR}/shifts"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] count check :: ls ${OUTPUT_DIR}/shifts/*.pt | wc -l == 18"
else
  n_pt="$(find "${OUTPUT_DIR}/shifts" -maxdepth 1 -name '*.pt' | wc -l)"
  n_mf="$(find "${OUTPUT_DIR}/shifts" -maxdepth 1 -name '*.manifest.json' | wc -l)"
  if [[ "$n_pt" != "18" || "$n_mf" != "18" ]]; then
    fail_loud 2 code "local_shift_count_mismatch_pt=${n_pt}_manifest=${n_mf}"
  fi
fi

# ── 7. upload BEFORE termination + fail-loud verify ──────────────────
phase upload "upload_folder ${OUTPUT_DIR}/shifts -> hf://${HF_DATA_REPO}/${HF_PREFIX}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] upload :: issue551_upload_verify.py (upload_folder + list_repo_files verify, 18 + 18)"
else
  UPLOAD_RC=0
  # Same code the VM-side smoke executed for real (round 2): extracted to a
  # standalone script so smoke and production share ONE upload/verify path.
  uv run python scripts/issue551_upload_verify.py \
    "$OUTPUT_DIR" "$HF_DATA_REPO" "$HF_PREFIX" || UPLOAD_RC=$?
  if (( UPLOAD_RC != 0 )); then
    # Plan §4 Step 4.7: pod KEPT ALIVE on upload-verify miss; the tensors
    # exist only on this pod until the upload lands.
    fail_loud 2 infra upload_verify_failed_POD_KEPT_ALIVE
  fi
fi

# ── 8. end-of-run results sentinel + [phase=done] ─────────────────────
phase write_sentinel "results sentinel for poll_pipeline.py"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] sentinel :: /workspace/logs/issue-551-epm_results-<epoch>.json"
else
  EPOCH="$(date +%s)"
  SENTINEL="/workspace/logs/issue-551-epm_results-${EPOCH}.json"
  uv run python - "$SENTINEL" "$OUTPUT_DIR" "$HF_DATA_REPO" "$HF_PREFIX" <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

sentinel, output_dir, repo_id, prefix = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
gate_path = output_dir / "smoke_gate_result.json"
gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
shift_files = sorted(p.name for p in (output_dir / "shifts").glob("*.pt"))
try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    git_commit = "unknown"

note = {
    "issue": 551,
    "phase": "extraction_complete_and_uploaded",
    "smoke_gate": gate,
    "n_shift_pt_files": len(shift_files),
    "shift_files_sample": shift_files[:6],
    "hf_tensor_prefix": f"{repo_id}/{prefix}",
    "git_commit": git_commit,
    "next_step": (
        "VM-side: uv run python scripts/issue551_controls.py "
        "--tensors-repo superkaiba1/explore-persona-space-data "
        "--tensors-prefix issue551_shift_reextract/analysis_tensors/shifts "
        "--parent-svd-dir eval_results/issue_521/svd --out eval_results/issue_551"
    ),
}
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 551,
    "by": "run_issue551_extract.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
with open(sentinel, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote sentinel: {sentinel}")
PY
fi

phase done "issue-551 extraction + persistence complete (controls run OFF-POD)"
