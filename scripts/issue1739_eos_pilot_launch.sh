#!/usr/bin/env bash
# #1739 evil-ood-spread pilot pod launcher: ONE corpus per pod.
#
#   usage: issue1739_eos_pilot_launch.sh <mhj|tomgibbs|pair> [extra args...]
#
# Thin env-hardening shell around scripts/issue1739_eos_pilot_pod.py (which owns
# contexts + generation + mirror + upload + the done sentinel). Mirrors the
# proven detached-launch pattern of issue1739_jobdr2aug_launch.sh.
set -euo pipefail

# A detached (setsid/nohup) launch inherits NO login PATH, so `uv` — installed
# at /root/.local/bin by bootstrap_pod.sh — is not found and the leg dies
# rc=127 seconds in (gotchas.md § setsid launcher PATH). FIRST thing done.
export PATH="/root/.local/bin:$PATH"

CORPUS="${1:?usage: issue1739_eos_pilot_launch.sh <mhj|tomgibbs|pair> [extra args...]}"
shift || true

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"

cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Hub accelerators, shell-level (frozen at import by huggingface_hub —
# upload-policy.md); a detached launch inherits no interactive profile.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
# vLLM V1 EngineCore dies silently under fork() when the parent touched
# CUDA-adjacent code first (gotchas.md #628); spawn is the safe default.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
mkdir -p "$LOG_DIR"

echo "[launch] eos-pilot corpus=$CORPUS commit=$(git rev-parse --short HEAD) extra=${*:-n/a}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
free -g | head -2

# NOT exec: this launcher owns the pod's SINGLE terminal [phase=done] line, so
# the child never emits the reserved token (#545/#920).
uv run python scripts/issue1739_eos_pilot_pod.py --corpus "$CORPUS" "$@"
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "[phase=done] eos pilot generation complete: corpus=$CORPUS"
fi
exit "$rc"
