#!/usr/bin/env bash
# Invoke only through the approved experiment task and monitored dispatcher.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EPS_STORY_PERSONA_OUT="${EPS_STORY_PERSONA_OUT:-/workspace/story_persona_qwen38}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTHONUNBUFFERED=1

: "${EPS_STORY_PERSONA_SOURCE_SHA:?dispatcher must pin the committed source SHA}"
: "${EPS_SENTINEL_PATH:?dispatcher must supply its completion sentinel path}"
if [[ ! "$EPS_STORY_PERSONA_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Invalid EPS_STORY_PERSONA_SOURCE_SHA' >&2
  exit 2
fi
mkdir -p "$EPS_STORY_PERSONA_OUT"
echo '[phase=preflight]'
uv run python -m explore_persona_space.orchestrate.preflight

# Qwen3.8 needs a newer runtime than the repository's default <5.0 pin.
# uv's existing overlay pattern leaves the project's locked environment intact.
echo '[phase=capture]'
uv run --with 'transformers==5.15.0' \
  python scripts/story_persona_qwen38_pilot.py phase=capture "$@"
echo '[phase=analyze]'
uv run --with 'transformers==5.15.0' \
  python scripts/story_persona_qwen38_pilot.py phase=analyze "$@"

# The artifact phase verifies all persisted bytes before writing both dispatcher
# completion channels. EPS_SENTINEL_PATH is supplied by the backend renderer.
echo '[phase=artifacts]'
uv run --with 'transformers==5.15.0' \
  python scripts/story_persona_qwen38_artifacts.py --phase publish \
  --out-dir "$EPS_STORY_PERSONA_OUT" --repo-root "$repo_root"
echo '[phase=done]'
