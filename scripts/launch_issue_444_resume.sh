#!/bin/bash
# RESUME launcher for task #444 — starts at fp-calibration.
#
# Use after the dataset + baselines phases have already completed (their outputs
# are on disk and reused): this skips fact-pick / dataset / baselines (which have
# no top-level skip and would otherwise redo ~25 min of on-policy generation +
# re-trigger the dataset guards). Runs:
#
#   fp-calibration -> 3 worker waves (seeds 42/137/256 x 4 conditions)
#     -> full-eval -> aggregate -> upload
#
# Idempotent: every phase caches/skip-resumes. fp-calibration reuses its cached
# verdicts JSONL (delete only fp_calibration_<slug>.json, NOT the verdicts, to
# re-run the aggregation under the non-blocking gate without re-judging).
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
export HF_HOME=/workspace/.cache/huggingface
mkdir -p /workspace/logs/issue-444
echo $$ > /workspace/logs/issue-444.pid
echo "[launcher] $(date -Is) RESUME-from-fp-calibration START HEAD=$(git rev-parse --short HEAD)"

phase () {  # $1 = banner label ; rest = args passed to the driver
  local label=$1; shift
  echo "[launcher] $(date -Is) [phase=${label}] START"
  if ! uv run python scripts/run_experiment_444.py "$@"; then
    echo "[launcher] $(date -Is) [phase=${label}] FAILED"
    exit 1
  fi
  echo "[launcher] $(date -Is) [phase=${label}] DONE"
}

CONDS=("no-contrast 0" "hand-written-contradictory-cn 1" "hand-written-suppression-cn 2" "on-policy-suppression-cn 3")
wave () {  # $1 = shard_id (wave index) ; $2 = seed
  local shard=$1 seed=$2
  echo "[launcher] $(date -Is) [phase=worker-seed${seed}] START"
  local pids=() conds=()
  for cg in "${CONDS[@]}"; do
    local cond=${cg% *} gpu=${cg#* }
    # CVD_PIN_EXEMPT: pre-#578 completed-task dispatcher kept verbatim; new launches must pin env CUDA_VISIBLE_DEVICES per cell (gotchas.md CVD-clobber)
    nohup uv run python scripts/run_experiment_444.py \
      --phase worker --shard-id "$shard" --num-shards 3 \
      --condition "$cond" --seed "$seed" --gpu-id "$gpu" \
      >> "/workspace/logs/issue-444/train_seed${seed}_${cond}.log" 2>&1 &
    pids+=("$!"); conds+=("$cond")
  done
  local fail=0 i=0
  for p in "${pids[@]}"; do
    if ! wait "$p"; then
      echo "[launcher] cell FAILED: ${conds[$i]} seed${seed} (see train_seed${seed}_${conds[$i]}.log)"
      fail=1
    fi
    i=$((i+1))
  done
  if [ "$fail" -ne 0 ]; then
    echo "[launcher] $(date -Is) [phase=worker-seed${seed}] FAILED"
    exit 1
  fi
  echo "[launcher] $(date -Is) [phase=worker-seed${seed}] DONE"
}

phase fp-calibration --phase fp-calibration --gpu-id 0
wave 0 42
wave 1 137
wave 2 256
phase full-eval --phase full-eval --gpu-id 0
phase aggregate --phase aggregate
phase upload    --phase upload

echo "[launcher] $(date -Is) [phase=done] FULL RUN COMPLETE"
