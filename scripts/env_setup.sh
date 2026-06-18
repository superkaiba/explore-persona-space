#!/bin/bash
set -euo pipefail
# Source this file to set up the environment for all scripts
# Usage: source scripts/env_setup.sh

# Derive project root from this script's location
PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

# Add workspace packages to PYTHONPATH
export PYTHONPATH=/workspace/pip_packages:${PYTHONPATH:-}

# Load API keys and config (safe: handles spaces, comments, and blank lines)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Set HuggingFace cache — /workspace/.cache/huggingface on RunPod (persistent,
# shared with all scripts and open-instruct). Falls back to the user-level
# shared cache ~/.cache/huggingface (NOT $PROJECT_ROOT/cache — project root is
# per-checkout, so every git worktree would grow its own multi-GB HF cache;
# mirrors env.py:_hf_home_default).
# RunPod discriminator (mirrors env.py:is_runpod_env): /workspace must be a
# real volume MOUNT (every pod mounts its volume there), or RUNPOD_POD_ID is
# set. A plain /workspace DIRECTORY — present on the dev VM since 2026-06-11
# (GCP-lane sentinel staging) and on GCE instances — routes as local.
# NOTE: Never use MED_OUTPUT_DIR here — it's an output dir, not a cache location.
if [ -n "${RUNPOD_POD_ID:-}" ] || mountpoint -q /workspace 2>/dev/null; then
    export HF_HOME="/workspace/.cache/huggingface"
    # Pip/uv temp + cache on the pod volume (pod root disk has no space).
    # Pod-only for the same reason as HF_HOME: on the dev VM these grew
    # uv lock/cache litter under the plain-dir /workspace.
    export TMPDIR=/workspace/tmp
    export PIP_CACHE_DIR=/workspace/tmp/pip_cache
else
    export HF_HOME="$HOME/.cache/huggingface"
fi

# Add CUDA and torch libs to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

# Confirm setup
echo "Environment configured:"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  PYTHONPATH includes /workspace/pip_packages"
echo "  HF_HOME=$HF_HOME"
echo "  ANTHROPIC_API_KEY set: $([ -n "$ANTHROPIC_API_KEY" ] && echo yes || echo no)"
