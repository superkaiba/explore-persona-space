#!/usr/bin/env bash
# Invoke only through the approved experiment task and monitored dispatcher.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EPS_STORY_PERSONA_OUT="${EPS_STORY_PERSONA_OUT:-/workspace/analysis_tensors_story_persona_qwen38_v2}"
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
# Preserve this custom out-root explicitly before the backend's crash handler runs.
# The default path also belongs to the GCP analysis_tensors* recovery sweep.
persist_on_error() {
  local failed_rc=$?
  trap - EXIT
  if (( failed_rc != 0 )); then
    echo "[failure-persist] workload exited with status $failed_rc"
    if ! timeout --kill-after=10s 600s uv run --with 'transformers==5.15.0' \
      python scripts/story_persona_qwen38_artifacts.py --phase persist-failure \
      --out-dir "$EPS_STORY_PERSONA_OUT" --repo-root "$repo_root"; then
      echo '[failure-persist] FAILED: preserve this compute disk for recovery' >&2
    fi
  fi
  exit "$failed_rc"
}
trap persist_on_error EXIT
echo '[phase=preflight]'
# Stage only the two refs required by full preflight; the original shallow clone
# otherwise attempted a history fetch that exceeded preflight's 90-second bound.
pilot_branch="$(git branch --show-current)"
test -n "$pilot_branch"
git config remote.origin.tagOpt --no-tags
timeout --kill-after=10s 240s git fetch --depth=1 --no-tags origin \
  "+refs/heads/$pilot_branch:refs/remotes/origin/$pilot_branch" \
  '+refs/heads/main:refs/remotes/origin/main'
test "$(git rev-parse HEAD)" = "$EPS_STORY_PERSONA_SOURCE_SHA"
uv run python -m explore_persona_space.orchestrate.preflight
uv run python - <<'PY'
import json
from pathlib import Path

meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
limits = [int(meminfo['MemTotal'].split()[0]) * 1024]
for name in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
    path = Path(name)
    if path.exists() and path.read_text().strip() != 'max':
        limits.append(int(path.read_text()))
effective = min(limits)
print(json.dumps({'preload_effective_ram_bytes': effective}), flush=True)
if effective < 100_000_000_000:
    raise RuntimeError('pilot requires at least 100 GB effective host/container RAM')
PY

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
