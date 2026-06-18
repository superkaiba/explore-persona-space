#!/usr/bin/env bash
# Issue #552 — CONTINUATION driver (round 3): Phase D -> end after the
# round-2 HF-quota crash at the tensor-upload checkpoint (Step 7b).
#
# Context: scripts/run_issue552_resume.sh completed Steps 0/R/4/4b/5/6/7
# (inverted gate PASS, raw completions on HF, adapters staged, Phase C
# production: all 9 shift tensors + manifests written 2026-06-10T15:01Z).
# Step 7b then 403'd: "exceeded your public storage space" on the LFS
# batch for superkaiba1/explore-persona-space-data — the SAME account-wide
# quota that killed the round-1 adapter persist. Tensor durability has
# been secured OUTSIDE this script (the #521 lost-tensor incident intent):
#   - VM copy: eval_results/issue_552/analysis_tensors_staging/
#     (sha256-verified identical to the pod copies), and
#   - WandB artifact issue552_analysis_tensors:v0
#     (https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/1ti6qczi).
# The HF data-repo upload (issue552_benign_control/analysis_tensors/) is
# DEFERRED to the uploader once quota frees — same contract as round 1.
#
# A full re-run of run_issue552_resume.sh is impossible by design: its
# resume preflight (rc=40/41/42) requires the sft_narrow_merged/ dirs that
# Step 4b legitimately deleted after the gate. This script therefore runs
# ONLY the remaining reviewed steps, copied verbatim from the resume script:
#
#   Step 7b'  tensor DURABILITY checkpoint (replaces the HF upload):
#             assert all 9 shifts/*.pt exist locally (Phase D inputs) and
#             record the HF deferral loudly. No HF round-trip.
#   Step 8    Phase D (SVD + dual nulls) + 9-file assert.   [verbatim]
#   sentinel + [phase=done]                                  [verbatim]
#
# Launch appends to the SAME log as the resume run so the
# poll_pipeline.py [phase=...] contract holds across the relaunch.
# This script NEVER shells out to scripts/task.py (CLAUDE.md rule).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
for key in HF_TOKEN WANDB_API_KEY; do
  if [[ -z "${!key:-}" ]]; then
    echo "[phase=fail] $(date -Is) env preamble: required env var $key is empty/unset" >&2
    exit 11
  fi
done
echo "[phase=env_ok] $(date -Is) HF_TOKEN/WANDB_API_KEY non-empty (OPENAI not needed: no judge phase remains)"

# Belt-and-braces: nothing here may attempt an HF upload while the
# account is over quota (the round-1 + round-2 403s).
unset EPM_PERSIST_ADAPTER_HF_REPO EPM_PERSIST_ADAPTER_SUBFOLDER
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

N_GPUS="${N_GPUS:-4}"
OUT_DIR="eval_results/issue_552"
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
# Step 7b' — tensor DURABILITY checkpoint (replaces the quota-blocked HF
# upload; durability already secured VM-side + WandB, see header).
# Phase D consumes the LOCAL tensors, so assert they are all present.
# ──────────────────────────────────────────────────────────────────────
phase tensor_durability_checkpoint "asserting 9 local shift tensors (HF upload DEFERRED: account over public-storage quota; durable via VM staging + WandB artifact issue552_analysis_tensors:v0)"
uv run python - <<'PY'
from pathlib import Path

src = Path("eval_results/issue_552/shifts")
expected = [
    f"{variant}_benign_seed{seed}.pt"
    for variant in ("same", "base", "on_policy")
    for seed in (42, 137, 256)
]
missing = [n for n in expected if not (src / n).exists()]
assert not missing, (
    f"TENSOR DURABILITY CHECKPOINT FAILED — missing local shift tensors: {missing}. "
    f"Refusing to proceed to Phase D (the #521 lost-tensor incident fix)."
)
print(f"tensor durability checkpoint PASS: all {len(expected)} local tensors present; "
      "HF upload deferred (quota), durable copies on VM + WandB")
PY

# ──────────────────────────────────────────────────────────────────────
# Step 8 — Phase D: SVD + dual nulls (1,000 reps) per (variant, seed); CPU, <15 min
# [verbatim from run_issue552_resume.sh]
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
# End-of-run sentinel (poll_pipeline.py contract) [verbatim]
# ──────────────────────────────────────────────────────────────────────
phase write_sentinel "writing end-of-run results sentinel"
uv run python scripts/issue552_write_sentinel.py --mode done \
  2>&1 | tee "$LOG_DIR/sentinel_done.log" || \
  fail_loud "$?" "results_sentinel_write_failed"

phase done "issue-552 benign-control continuation complete (Phase D -> end after the round-2 HF-quota crash; tensor HF upload deferred to the uploader)"
