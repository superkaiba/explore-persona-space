#!/usr/bin/env bash
set -euo pipefail

# Round-8 GPU leg for issue #2254.  The owning VM session launches this file
# detached, verifies its HF artifacts, and terminates the pod before judging.

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
OUT_ROOT="${EPM_REVMAP8_OUT_ROOT:-${REPO_ROOT}/eval_results/issue_2254}"
LOG_ROOT="/workspace/logs"
PID_FILE="${LOG_ROOT}/issue-2254.pid"
RUN_STAMP="${EPM_REVMAP8_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
WORKER_LOG_ROOT="${LOG_ROOT}/issue-2254-revmap8gpu-workers-${RUN_STAMP}"
DRIVER="scripts/issue2254_revmap_dose_patch.py"

mkdir -p "$LOG_ROOT" "$WORKER_LOG_ROOT"
printf '%s\n' "$$" > "${PID_FILE}.tmp"
mv "${PID_FILE}.tmp" "$PID_FILE"

cd "$REPO_ROOT"
if [[ -f ./.env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
export PATH="/root/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2

write_progress_sentinel() {
  local commit epoch target
  commit=$(git rev-parse HEAD)
  epoch=$(date +%s)
  target="${LOG_ROOT}/issue-2254-epm_revmap8_gpu_leg-${epoch}.json"
  printf '{"sentinel_schema_version":1,"kind":"epm:revmap8-gpu-leg","version":1,"task_id":2254,"gate":"revmap8-gpu-leg","blocks_pipeline":false,"by":"codex-revmap8-owner","ts":%s,"note":"round=revmap_dose_patch leg=gpu-generation status=done commit=%s hf_prefix=issue2254_preimage/revmap_dose_patch"}\n' \
    "$epoch" "$commit" > "${target}.tmp"
  mv "${target}.tmp" "$target"
}

run_wave() {
  local phase="$1"
  local gpu rc failed
  local -a pids=()
  local -a logs=()
  failed=0
  echo "[phase=${phase}] launching four CVD-pinned shards"
  for gpu in 0 1 2 3; do
    logs+=("${WORKER_LOG_ROOT}/${phase}-shard${gpu}.log")
    env CUDA_VISIBLE_DEVICES="$gpu" UV_NO_SYNC=1 \
      timeout --kill-after=30s 10800 \
      uv run python "$DRIVER" \
        --phases "$phase" \
        --out-root "$OUT_ROOT" \
        --shard-id "$gpu" \
        --num-shards 4 \
        > "${logs[$gpu]}" 2>&1 &
    pids+=("$!")
    echo "[${phase}] shard=${gpu} pid=${pids[$gpu]} log=${logs[$gpu]}"
  done
  for gpu in 0 1 2 3; do
    set +e
    wait "${pids[$gpu]}"
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      failed=1
      echo "[${phase}] shard=${gpu} failed rc=${rc}; inner-log tail follows"
      tail -n 120 "${logs[$gpu]}" || true
    else
      echo "[${phase}] shard=${gpu} complete"
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi
}

echo "[phase=preflight] production commit=$(git rev-parse HEAD) out_root=${OUT_ROOT}"
timeout --kill-after=30s 1800 uv run python -m explore_persona_space.orchestrate.preflight
timeout --kill-after=10s 60 uv run python -c \
  'import huggingface_hub, torch, transformers; import scripts.issue2254_revmap_dose_patch'
uv run python "$DRIVER" --import-check

mapfile -t GPU_USED < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
if [[ "${#GPU_USED[@]}" -lt 4 ]]; then
  echo "expected at least four GPUs, found ${#GPU_USED[@]}" >&2
  exit 2
fi
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader || true
for gpu in 0 1 2 3; do
  used="${GPU_USED[$gpu]//[[:space:]]/}"
  if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > 2048 )); then
    echo "GPU ${gpu} is not clean before launch: memory.used=${used} MiB" >&2
    exit 3
  fi
done

echo "[phase=calibrate] capture neutral and positive operating projections"
env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 \
  timeout --kill-after=30s 7200 \
  uv run python "$DRIVER" --phases calibrate --out-root "$OUT_ROOT"

run_wave steer
run_wave patch

echo "[phase=upload_verify] generation checkpoints were packed and uploaded per shard"
write_progress_sentinel
echo "[phase=done]"
