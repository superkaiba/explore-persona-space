#!/bin/bash
# Full downstream run for task #444 (post fact-pick id=6).
#
# Phase order is the AUTHORITATIVE one from the PHASES tuple + the
# "fp-calibration now runs AFTER baselines" code comment — NOT the stale
# plan §10 nohup snippet (which mis-ordered fp-calibration before
# fact-candidates and omitted it downstream; that mis-order caused the
# original 2026-06-02 crash).
#
#   fact-pick -> dataset -> baselines -> fp-calibration
#     -> worker waves (3 seeds x 4 conditions, 4 GPUs/wave)
#     -> full-eval (vLLM generate + inline synchronous Haiku judge)
#     -> aggregate -> upload
#
# Idempotent: every phase caches its output and skips on re-run, so a crash
# mid-chain is fixed + this script re-launched to RESUME from the failed phase.
# `phase()` exits non-zero on any phase failure so the chain stops before the
# expensive waves if an upstream phase breaks.
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
export HF_HOME=/workspace/.cache/huggingface
mkdir -p /workspace/logs/issue-444
echo $$ > /workspace/logs/issue-444.pid
echo "[launcher] $(date -Is) full-run START HEAD=$(git rev-parse --short HEAD)"

phase () {  # $1 = banner label ; rest = args passed to the driver
  local label=$1; shift
  echo "[launcher] $(date -Is) [phase=${label}] START"
  if ! uv run python scripts/run_experiment_444.py "$@"; then
    echo "[launcher] $(date -Is) [phase=${label}] FAILED"
    exit 1
  fi
  echo "[launcher] $(date -Is) [phase=${label}] DONE"
}

phase fact-pick      --phase fact-pick --fact-pick-id 6
phase dataset        --phase dataset --gpu-id 0
phase baselines      --phase baselines --gpu-id 0
phase fp-calibration --phase fp-calibration --gpu-id 0

# Worker waves: 3 sequential waves (one per seed), 4 conditions in parallel
# (one GPU lane each). shard-id = wave index (matches the plan's worker call).
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
wave 0 42
wave 1 137
wave 2 256

phase full-eval --phase full-eval --gpu-id 0
phase aggregate --phase aggregate
phase upload    --phase upload

echo "[launcher] $(date -Is) [phase=done] FULL RUN COMPLETE"
