#!/usr/bin/env bash
# RunPod-only sequential GPU phases; the owning VM verifies and terminates the pod.
set -euo pipefail
umask 077
export PATH="/root/.local/bin:$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

repo=/workspace/explore-persona-space
run_root=/workspace/issue952-china-repair-v2
expected_sha="${1:?Pass the reviewed immutable code SHA}"
cd "$repo"
test "$(git rev-parse HEAD)" = "$expected_sha"
git merge-base --is-ancestor "$expected_sha" HEAD
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked source is dirty; refusing to run" >&2
  exit 86
fi
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

mkdir -p "$run_root/logs" /workspace/logs
printf '%s\n' "$$" > /workspace/logs/issue-952-china-repair-v2.pid
uv run --no-sync python -m explore_persona_space.orchestrate.preflight

run_phase() {
  local mode="$1"
  local phase="$2"
  shift 2
  local phase_log="$run_root/logs/${mode}-${phase}-$(date -u +%Y%m%dT%H%M%S).log"
  echo "[phase=${mode}_${phase//-/_}] code_sha=$expected_sha log=$phase_log"
  if [[ "$phase" = gen || "$phase" = capture ]]; then
    uv run --no-sync python -c \
      'import sys; from explore_persona_space.orchestrate.preflight import assert_out_root_headroom; assert_out_root_headroom(sys.argv[1], 20, phase=sys.argv[2])' \
      "$run_root" "${mode}_${phase}"
  fi
  uv run --no-sync python -m scripts.issue952_china_definitive_gpu \
    --study repaired-v2 --phase "$phase" --out-root "$run_root/$mode" \
    --attempt 1 "$@" > "$phase_log" 2>&1
  echo "Completed ${mode}/${phase} rc=0"
}

run_phase smoke gen --smoke
run_phase smoke upload-raw
run_phase smoke capture --smoke
run_phase smoke upload-capture
run_phase smoke finalize
run_phase production gen --smoke-report "$run_root/smoke/manifests/smoke_timing.json"
run_phase production upload-raw
run_phase production capture --smoke-report "$run_root/smoke/manifests/smoke_timing.json"
run_phase production upload-capture
run_phase production finalize
test -s "$run_root/production/issue952_china_definitive_done.json"
echo "[phase=done] GPU work and phase uploads finished; VM owner must verify out-root and teardown"
