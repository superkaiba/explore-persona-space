#!/usr/bin/env bash
# Issue #366 cascade experiment pod entrypoint.
#
# Invoked by Sagan's bootstrap wrapper via dockerArgs:
#     bash -lc 'set -euo pipefail; cd /workspace && bash scripts/experiments/366/run_366.sh'
#
# The wrapper has already done: clone repo onto branch $SAGAN_EPS_BRANCH at
# /workspace, installed deps via `uv sync`, redirected HF_HOME / WANDB_*
# caches to /workspace, and posted 5% progress. Our job is to run the
# Python orchestrator and let the wrapper post the final 100% on exit.

set -euo pipefail

# Always work from the repo root regardless of how this script was invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================================="
echo "Issue #366 — cross-persona chunk-binding cascade"
echo "Repo:           $REPO_ROOT"
echo "Branch (env):   ${SAGAN_EPS_BRANCH:-unknown}"
echo "Commit (env):   ${SAGAN_EPS_COMMIT_SHA:-unknown}"
echo "Git HEAD:       $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Date:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HF_HOME:        ${HF_HOME:-unset}"
echo "WANDB_PROJECT:  ${WANDB_PROJECT:-unset}"
echo "=================================================================="

# Default WANDB_PROJECT if the wrapper didn't set one.
export WANDB_PROJECT="${WANDB_PROJECT:-issue366_cascade}"
# vLLM memory budget: 0.60 matches #354 and leaves headroom for the merge
# step's transient memory spike. Override via env if your pod has different
# constraints.
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.60}"

# Pin Python to the one uv installed.
exec uv run python scripts/experiments/366/run_366.py --gpu 0
