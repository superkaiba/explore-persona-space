#!/usr/bin/env bash
# Issue #641 (Phase 2) — dispatch wrapper with HF debug-log upload on EXIT.
#
# The base GCE startup script powers off the VM on rc!=0 (EXIT trap), and
# eps-router has no logging.read IAM. Without an external log surface we are
# fully blind on workload failure. This wrapper uploads its own log to the HF
# dataset repo on EXIT so post-mortem diagnosis is possible.
set -euo pipefail

# --- Logging setup BEFORE anything else ---
DEBUG_LOG="/workspace/logs/issue-641-dispatch-debug.log"
mkdir -p /workspace/logs
exec > >(tee -a "$DEBUG_LOG") 2>&1
set -x  # trace every command for full visibility

# --- EXIT trap: upload debug log to HF before VM dies ---
on_exit() {
    local rc=$?
    set +x
    echo "[issue641-dispatch] EXIT rc=$rc cwd=$(pwd) at $(date -u +%FT%TZ)"
    echo "[issue641-dispatch] env: REPO_ROOT=${REPO_ROOT:-unset} WORKLOAD_ROOT=${WORKLOAD_ROOT:-unset} HF_TOKEN_set=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no)"
    if [ -f "$DEBUG_LOG" ] && [ -n "${HF_TOKEN:-}" ]; then
        local ts; ts=$(date -u +%Y%m%dT%H%M%SZ)
        local dst="issue641_debug/dispatch-rc${rc}-${ts}.log"
        echo "[issue641-dispatch] uploading $DEBUG_LOG -> hf://${dst}"
        uv run python - <<PY 2>&1 || echo "[issue641-dispatch] HF upload failed: $?"
import os
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj="$DEBUG_LOG",
    path_in_repo="$dst",
    repo_id="superkaiba1/explore-persona-space-data",
    repo_type="dataset",
    commit_message="#641 dispatch debug rc=$rc",
)
print(f"uploaded -> hf://$dst")
PY
    else
        echo "[issue641-dispatch] skipping HF upload (log_exists=$([ -f "$DEBUG_LOG" ] && echo yes || echo no), HF_TOKEN=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset))"
    fi
    return $rc
}
trap on_exit EXIT

# --- Now the real dispatch ---
cd "${REPO_ROOT:-/workspace/explore-persona-space}"
export TQDM_DISABLE=1
DISPATCH="scripts/issue641_dose_curves.py"
LADDER="50,100,150,250,375,560"

echo "[issue641-dispatch] starting at $(date -u +%FT%TZ) repo=$(pwd)"
echo "[issue641-dispatch] env check:"
which uv && uv --version
which python || true
ls -la "$DISPATCH"
echo "[issue641-dispatch] --- begin pipeline ---"

# 1. P0 — base-model harmful-advice propensity
echo "[issue641-dispatch] PHASE: base-propensity at $(date -u +%FT%TZ)"
uv run python "$DISPATCH" --phase base-propensity --seeds 42

# 2. Arm A — 6 #537 EM source contexts x 2 seeds, dose ladder
echo "[issue641-dispatch] PHASE: run Arm A at $(date -u +%FT%TZ)"
uv run python "$DISPATCH" --phase run \
    --sources icl_k2,wc_short_advice,sp_doctor,reph_imp,sp_ph1,wc_short_code \
    --seeds 42,1042 --max-steps 560 --save-steps 25 --save-total-limit 30 \
    --ladder "$LADDER" --probes 8 --samples 5

# 3. select-neutral
echo "[issue641-dispatch] PHASE: select-neutral at $(date -u +%FT%TZ)"
uv run python "$DISPATCH" --phase select-neutral

NEUTRAL="$(uv run python -c "import json,os; p=os.path.join('eval_results/issue_641','base_propensity','matched_neutral.json'); print(json.load(open(p))['persona_key'])")"
if [ -z "$NEUTRAL" ]; then
    echo "[issue641-dispatch] FATAL: matched_neutral.json missing persona_key" >&2
    exit 1
fi
echo "[issue641-dispatch] matched neutral = ${NEUTRAL}"

# 4. Arm B
echo "[issue641-dispatch] PHASE: run Arm B (sp_teacher_ho,${NEUTRAL}) at $(date -u +%FT%TZ)"
uv run python "$DISPATCH" --phase run \
    --sources "sp_teacher_ho,${NEUTRAL}" \
    --seeds 42,1042 --max-steps 560 --save-steps 25 --save-total-limit 30 \
    --ladder "$LADDER" --probes 8 --samples 5

echo "[issue641-dispatch] GPU pipeline complete at $(date -u +%FT%TZ) (P4 aggregate runs off-pod on the VM)"
