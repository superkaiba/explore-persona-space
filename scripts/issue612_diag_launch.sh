#!/usr/bin/env bash
# Diagnostic launch wrapper for issue #612 v3 amendment.
#
# 1. Run prefetch_inputs for ALL 16 v3 cells (the driver's Phase A bypasses
#    the dispatcher's prefetch; pools_411/* must be on local disk before
#    build_predictor_v3_pool can call parse_frozen_pool).
# 2. Run the driver, tee output to /workspace/driver.log.
# 3. On EXIT (any rc), upload driver.log + every phase_log to HF data repo.

set -uo pipefail   # NB: no -e — we want diagnostic upload even on driver failure

WORKDIR="${WORKLOAD_ROOT:-/workspace/eps-issue-612}"
DRIVER_LOG=/workspace/driver.log
DIAG_DIR=/workspace/diagnostic
HFREPO="superkaiba1/explore-persona-space-data"
RUN_TAG="${ISSUE612_DIAG_RUN_TAG:-run-$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$WORKDIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

upload_diag() {
  mkdir -p "$DIAG_DIR"
  cp "$DRIVER_LOG" "$DIAG_DIR/driver.log" 2>/dev/null || true
  find "$WORKDIR/data" "$WORKDIR/eval_results" "$WORKDIR/logs" -name "*.log" -type f 2>/dev/null | while read -r f; do
    rel=$(echo "$f" | sed "s|$WORKDIR/||" | tr '/' '__')
    cp "$f" "$DIAG_DIR/${rel}" 2>/dev/null || true
  done
  ls -la "$DIAG_DIR/"
  tar czf /workspace/diag.tgz -C /workspace diagnostic 2>/dev/null
  uv run python -c "
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj=open('/workspace/diag.tgz', 'rb'),
    path_in_repo='issue612_onpolicy_leakage_predictor/diagnostic/${RUN_TAG}.tgz',
    repo_id='${HFREPO}',
    repo_type='dataset',
    commit_message='issue-612 diagnostic ${RUN_TAG}',
)
print('diagnostic uploaded: ${RUN_TAG}.tgz')
" 2>&1 || echo "diag upload failed"
}

trap upload_diag EXIT

# Prefetch ALL 16 v3 cells' frozen inputs BEFORE driver runs.
# (The driver's Phase A invokes build_predictor_v3_pool directly, which calls
# parse_frozen_pool expecting pools_411/<source>_seed42/train_pool.jsonl on
# local disk; the dispatcher's normal --stage predictor-v3 prefetch is BYPASSED
# by Phase A. So pre-fetch manually here.)
ALL_CELLS="villain:arm_canned:42,villain:arm_canned:137,villain:arm_onpolicy:42,villain:arm_onpolicy:137,comedian:arm_canned:42,comedian:arm_canned:137,comedian:arm_onpolicy:42,comedian:arm_onpolicy:137,kindergarten_teacher:arm_canned:42,kindergarten_teacher:arm_canned:137,kindergarten_teacher:arm_onpolicy:42,kindergarten_teacher:arm_onpolicy:137,software_engineer:arm_canned:42,software_engineer:arm_canned:137,software_engineer:arm_onpolicy:42,software_engineer:arm_onpolicy:137"
echo "[diag_launch] prefetch_inputs for 16 v3 cells..."
uv run python -m explore_persona_space.experiments.sycophancy_onpolicy_612.prefetch_inputs --cells "$ALL_CELLS" 2>&1 | tee "$DRIVER_LOG"
prefetch_rc=${PIPESTATUS[0]}
if [ "$prefetch_rc" -ne 0 ]; then
  echo "=== PREFETCH FAILED rc=$prefetch_rc ==="
  exit "$prefetch_rc"
fi

echo "[diag_launch] launching v3 driver..."
uv run python scripts/issue612_predictor_v3_driver.py --gpus 0,1,2,3 2>&1 | tee -a "$DRIVER_LOG"
exit "${PIPESTATUS[0]}"
