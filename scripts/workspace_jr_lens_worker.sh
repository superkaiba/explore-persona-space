#!/usr/bin/env bash
# One disjoint calibration interval on one visible GPU; no main outcome reads.
set -euo pipefail
set +x
JR_RANK=${1:?rank required}
JR_START=${2:?start required}
JR_STOP=${3:?stop required}
cd /workspace/explore-persona-space
[[ $(git rev-parse HEAD) == 0f23250df469235c8dad70b86cd93b7b4f3c318a ]]
[[ -z $(git status --porcelain --untracked-files=no) ]]
JR_OUT=/workspace/workspace_jr/primary_full_calibration
JR_EXIT="$JR_OUT/rank${JR_RANK}_exit.json"
[[ ! -e "$JR_EXIT" ]]
trap 'JR_RC=$?; printf "{\"exit_code\":%d,\"rank\":%d,\"start\":%d,\"stop\":%d,\"finished_at_epoch\":%d}\n" "$JR_RC" "$JR_RANK" "$JR_START" "$JR_STOP" "$(date +%s)" > "$JR_EXIT"' EXIT
export CUDA_VISIBLE_DEVICES="$JR_RANK"
export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2
/root/.local/bin/uv run --project runtime/workspace_jr --frozen scripts/workspace_jr_runtime.py \
  --role primary fit-lens-shard --token-manifest "$JR_OUT/calibration_tokens.json" \
  --validation "$JR_OUT/native_validation.json" --out-dir "$JR_OUT/lens_shards" \
  --start "$JR_START" --stop "$JR_STOP" --dim-batch 8 --device cuda:0
