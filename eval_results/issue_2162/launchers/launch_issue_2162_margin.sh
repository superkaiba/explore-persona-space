#!/bin/bash
# Deferred TF-margin leg for issue #2162: dispatch.sh margin && dispatch.sh upload.
# Runs CONCURRENTLY with the P7 CPU analysis chain (wrapper pid 3026) on this pod;
# distinct log/pidfile namespace (issue-2162-margin.*). Never touches P7 state.
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && . ./.env; set +a
export EPM_2162_GATE3=/workspace/explore-persona-space/eval_results/issue_2162/judge/gates/separation_gate_report.json
echo $$ > /workspace/logs/issue-2162-margin.pid
bash scripts/issue2162_dispatch.sh margin && bash scripts/issue2162_dispatch.sh upload
rc=$?
echo "[margin-leg] chain exit rc=$rc"
exit $rc
