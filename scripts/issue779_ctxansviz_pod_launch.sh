#!/bin/bash
# Pod-side detached launcher for the issue #779 ctxansviz pipeline (inline viz round).
# Canonical setsid launcher-script shape (.claude/agents/experimenter.md § During
# Execution): the SSH shell is sh, so env-sourcing lives HERE (bash), never at the
# SSH top level. Launch from the pod as:
#   setsid nohup bash /workspace/explore-persona-space/scripts/issue779_ctxansviz_pod_launch.sh \
#     > /workspace/logs/issue-779-ctxansviz.log 2>&1 < /dev/null &
# Extra args (e.g. --smoke) are forwarded to the python driver.
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a
[ -f .env ] && source .env
set +a
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Dedicated 16-vCPU CPU pod: full-width BLAS threads + the glibc arena cap.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1
mkdir -p /workspace/logs
# $$ becomes the python PID after the exec chain (shell -> uv run -> python).
echo $$ > /workspace/logs/issue-779-ctxansviz.pid
exec uv run python scripts/issue779_ctxansviz_pod.py --phase all "$@"
