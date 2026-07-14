#!/usr/bin/env bash
# #1090 fu3 (posonly-contexts-parallel-matrix) pod-side launcher (plan v5 §D7).
#
# Thin shell: env sourcing + pid-file rewrite + the python work-conserving
# dispatcher (scripts/issue1090_fu3_worker.py dispatch — per-slot
# CUDA_VISIBLE_DEVICES + VLLM_PORT=8000+i pins live in the python launcher's
# child env). [phase=done] is emitted HERE, only after the dispatcher exits 0
# (pod-side-reporting.md: the token is reserved for this single terminal line;
# per-cell worker logs are redirected to per-cell files by the dispatcher).
#
# Launch (experimenter, detached):
#   setsid nohup bash scripts/issue1090_fu3_dispatch.sh > /workspace/logs/issue-1090-fu3.log 2>&1 < /dev/null &
# Smoke parity: same script, subsetted — e.g.
#   bash scripts/issue1090_fu3_dispatch.sh --cells C3-bare-pos --n-gpus 1 --smoke
set -euo pipefail

cd "$(dirname "$0")/.."

# GCE lane exports tokens via startup metadata and has NO .env — source conditionally.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${EPM_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"
# Pid-file contract (pod-side-reporting.md): rewrite on EVERY (re)launch.
echo $$ > "$LOG_DIR/issue-1090.pid"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

uv run python scripts/issue1090_fu3_worker.py dispatch --sentinel-dir "$LOG_DIR" "$@"

echo "[phase=done]"
