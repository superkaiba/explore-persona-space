#!/usr/bin/env bash
# Issue #641 follow-up — identity-conflict-more-seeds. Arm-B-only re-run at
# parent-identical schedule (max_steps=560) with 6 NEW seeds for the 8-seed
# Arm-B ΔL hierarchical-bootstrap at the matched dose (step 100). The parent's
# 4 step-100 cells (teacher/historian × {42, 1042}) are pooled in the off-pod
# aggregate phase via --extra-records-roots.
set -euo pipefail

DEBUG_LOG="/workspace/logs/issue-641-more-seeds-dispatch-debug.log"
mkdir -p /workspace/logs
exec > >(tee -a "$DEBUG_LOG") 2>&1
set -x

on_exit() {
    local rc=$?
    set +x
    echo "[issue641-more-seeds] EXIT rc=$rc cwd=$(pwd) at $(date -u +%FT%TZ)"
    echo "[issue641-more-seeds] env: REPO_ROOT=${REPO_ROOT:-unset} WORKLOAD_ROOT=${WORKLOAD_ROOT:-unset} HF_TOKEN_set=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no)"
    if [ -f "$DEBUG_LOG" ] && [ -n "${HF_TOKEN:-}" ]; then
        local ts; ts=$(date -u +%Y%m%dT%H%M%SZ)
        local dst="issue641_debug/more_seeds_dispatch-rc${rc}-${ts}.log"
        echo "[issue641-more-seeds] uploading $DEBUG_LOG -> hf://${dst}"
        uv run python - <<PY 2>&1 || echo "[issue641-more-seeds] HF upload failed: $?"
import os
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj="$DEBUG_LOG",
    path_in_repo="$dst",
    repo_id="superkaiba1/explore-persona-space-data",
    repo_type="dataset",
    commit_message="#641 more-seeds dispatch debug rc=$rc",
)
print(f"uploaded -> hf://$dst")
PY
    else
        echo "[issue641-more-seeds] skipping HF upload"
    fi
    return $rc
}
trap on_exit EXIT

cd "${REPO_ROOT:-/workspace/explore-persona-space}"
DISPATCH="scripts/issue641_dose_curves.py"
# Identity-conflict pair: teacher + matched neutral (local_historian, gap 0.0
# per parent's matched_neutral.json — committed on issue-641).
NEUTRAL="$(uv run python -c "import json,os; p=os.path.join('eval_results/issue_641','base_propensity','matched_neutral.json'); print(json.load(open(p))['persona_key'])")"
if [ -z "$NEUTRAL" ]; then
    echo "[issue641-more-seeds] FATAL: matched_neutral.json missing persona_key" >&2
    exit 1
fi
echo "[issue641-more-seeds] matched neutral = ${NEUTRAL}"

# Route writes to the follow-up artifact dir; parent's dose_curves/ unchanged.
export I641_EVAL_ROOT="$PWD/eval_results/issue_641/identity-conflict-more-seeds"
mkdir -p "$I641_EVAL_ROOT"

# Arm B only, parent's identical schedule (max_steps=560 + save_steps=25), 6
# new seeds, eval ONLY at step 100 via --ladder 100. The schedule-parity assert
# in phase_aggregate (off-pod, on the VM) verifies all 8 pooled cells share
# max_steps=560 / linear before computing the headline ΔL.
echo "[issue641-more-seeds] PHASE: run Arm B more-seeds at $(date -u +%FT%TZ)"
# save_total_limit must be >= ceil((max_steps - min(ladder)) / save_steps) + 1
# + safety margin, or HF Trainer prunes "to last N" and silently deletes the
# step-100 ladder checkpoint long before training ends (the post-train ladder
# read then fails). Parent uses 30 (sufficient for max_steps=560 + save_steps=25
# + ladder rungs as low as 50); match it. (round-3 fix; in-script floor in
# _train_dose_ladder is the defense-in-depth backstop.)
uv run python "$DISPATCH" --phase run \
    --sources "sp_teacher_ho,${NEUTRAL}" \
    --seeds 1,7,123,2024,31337,98765 \
    --max-steps 560 --save-steps 25 --save-total-limit 30 \
    --ladder 100 --probes 8 --samples 5

echo "[issue641-more-seeds] GPU pipeline complete at $(date -u +%FT%TZ); aggregate runs off-pod on the VM"
