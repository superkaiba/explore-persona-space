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
# shared with all scripts and open-instruct). Falls back to project-local cache.
# NOTE: Never use MED_OUTPUT_DIR here — it's an output dir, not a cache location.
if [ -d "/workspace" ]; then
    export HF_HOME="/workspace/.cache/huggingface"
else
    export HF_HOME="$PROJECT_ROOT/cache/huggingface"
fi

# Add CUDA and torch libs to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

# Set pip temp dir to workspace (root has no space)
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/tmp/pip_cache

# Confirm setup
echo "Environment configured:"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  PYTHONPATH includes /workspace/pip_packages"
echo "  HF_HOME=$HF_HOME"
echo "  ANTHROPIC_API_KEY set: $([ -n "$ANTHROPIC_API_KEY" ] && echo yes || echo no)"
