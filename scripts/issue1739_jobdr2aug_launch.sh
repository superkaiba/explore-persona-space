#!/usr/bin/env bash
# #1739 jobd/r2aug pod-side launcher: one (behavior x job) unit per pod.
#
#   usage: issue1739_jobdr2aug_launch.sh <behavior> <jobd|r2aug> [conditions...]
#
# Thin env-hardening shell around scripts/issue1739_jobd_r2aug_run.py (which
# owns staging + canaries + scoring + upload + the done sentinel). Mirrors the
# proven detached-launch pattern of issue1768_dyn_launch.sh.
set -euo pipefail

# A detached (setsid/nohup) launch inherits NO login PATH, so `uv` — installed
# at /root/.local/bin by bootstrap_pod.sh — is not found and the leg dies
# rc=127 seconds in (gotchas.md § setsid launcher PATH). FIRST thing done.
export PATH="/root/.local/bin:$PATH"

BEHAVIOR="${1:?usage: issue1739_jobdr2aug_launch.sh <behavior> <jobd|r2aug> [conditions...]}"
JOB="${2:?usage: issue1739_jobdr2aug_launch.sh <behavior> <jobd|r2aug> [conditions...]}"
shift 2
CONDITIONS=("$@")

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"

cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Hub accelerators, shell-level (frozen at import by huggingface_hub —
# upload-policy.md); a detached launch inherits no interactive profile.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
mkdir -p "$LOG_DIR"

echo "[launch] behavior=$BEHAVIOR job=$JOB conditions=${CONDITIONS[*]:-n/a} \
commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
free -g | head -2

ARGS=(--behavior "$BEHAVIOR" --modes "$JOB")
if [ "$JOB" = "r2aug" ] && [ "${#CONDITIONS[@]}" -gt 0 ]; then
  ARGS+=(--map-conditions "${CONDITIONS[@]}")
fi
exec uv run python scripts/issue1739_jobd_r2aug_run.py "${ARGS[@]}"
