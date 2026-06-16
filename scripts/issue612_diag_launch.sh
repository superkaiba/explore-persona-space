#!/usr/bin/env bash
# Diagnostic launch wrapper for issue #612 v3 amendment.
#
# Wraps the driver, tees ALL output to /workspace/driver.log, and on EXIT
# (regardless of rc) bundles + uploads to HF data repo the driver.log + every
# phase_log (build_predictor_v3_pool's pool_build.log, eval_panel logs, any
# *.log under data/issue_612 + eval_results/issue_612). This is a one-time
# diagnostic harness for tasks #612 v3 relaunches that keep dying mid-Phase-A
# with serial-port output inaccessible post-TERMINATED.
#
# The driver's _run() redirects each subprocess's stdout+stderr to a per-phase
# log FILE on the instance disk — those die with the instance unless we
# explicitly upload them. This script does that.

set -uo pipefail   # NB: no -e — we want to upload diagnostics even on driver failure

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
  # Phase logs
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

uv run python scripts/issue612_predictor_v3_driver.py --gpus 0,1,2,3 2>&1 | tee "$DRIVER_LOG"
exit "${PIPESTATUS[0]}"
