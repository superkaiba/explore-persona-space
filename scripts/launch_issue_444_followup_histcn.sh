#!/bin/bash
# Inline FOLLOW-UP launcher for task #444 — adds `local_historian` as a
# contrastive negative in the on-policy-suppression arm ONLY.
#
# Single-variable delta from the parent run:
#   - On-policy CN training personas: 4 -> 5 (+ local_historian).
#   - Conditions trained: 1 (on-policy-suppression-cn) x 3 seeds = 3 cells.
#   - `local_resident` stays eval-only (held-out content-fit control).
#   - Eval frame unchanged (7 personas, incl. local_historian + local_resident).
#
# Output paths are re-rooted under `local_historian_as_cn/` so the parent's
# 12-cell artifacts are NEVER touched:
#   eval_results/issue_444/local_historian_as_cn/
#   data/exp444/local_historian_as_cn/
#   outputs/exp444_adapters/local_historian_as_cn/
#   figures/issue_444/local_historian_as_cn/
# HF data-repo bucket + WandB project + adapter HF path are also namespaced.
#
# Phase order (mirrors the parent's full-run launcher; fact-pick is reused
# from the parent via file copy — fact-candidates / fact-pick involve a
# user gate + Sonnet calls and re-running them would change the picked
# (figure, attribute) pair, breaking the single-variable contract):
#
#   (copy parent's fact_pick.json into the follow-up namespace)
#   -> dataset -> baselines -> fp-calibration
#   -> worker wave (3 seeds in parallel; 1 condition; 1 GPU per seed)
#   -> full-eval (vLLM generate + Haiku judge) -> aggregate -> upload
#
# Idempotent: every phase caches its output and skips on re-run.
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
export HF_HOME=/workspace/.cache/huggingface
# Gate ALL follow-up routing in the driver. Without this env var the driver
# is byte-for-byte equivalent to the parent.
export EPM_444_FOLLOWUP_HISTORIAN_CN=1
mkdir -p /workspace/logs/issue-444
echo $$ > /workspace/logs/issue-444-followup-histcn.pid
echo "[launcher] $(date -Is) FOLLOWUP-histcn START HEAD=$(git rev-parse --short HEAD)"
echo "[launcher] EPM_444_FOLLOWUP_HISTORIAN_CN=${EPM_444_FOLLOWUP_HISTORIAN_CN}"

# Copy the parent's fact_pick.json into the follow-up namespace so the
# single-variable contract holds (same figure, same invented attribute).
PARENT_PICK=eval_results/issue_444/phase0_fact_candidates/fact_pick.json
FOLLOWUP_PICK=eval_results/issue_444/local_historian_as_cn/phase0_fact_candidates/fact_pick.json
if [ ! -f "$PARENT_PICK" ]; then
  echo "[launcher] FAILED: parent fact_pick.json missing at $PARENT_PICK; run the parent's fact-pick first"
  exit 1
fi
mkdir -p "$(dirname "$FOLLOWUP_PICK")"
if [ ! -f "$FOLLOWUP_PICK" ]; then
  cp "$PARENT_PICK" "$FOLLOWUP_PICK"
  echo "[launcher] copied $PARENT_PICK -> $FOLLOWUP_PICK"
else
  echo "[launcher] $FOLLOWUP_PICK already exists; not overwriting (idempotent)"
fi

phase () {  # $1 = banner label ; rest = args passed to the driver
  local label=$1; shift
  echo "[launcher] $(date -Is) [phase=${label}] START"
  if ! uv run python scripts/run_experiment_444.py "$@"; then
    echo "[launcher] $(date -Is) [phase=${label}] FAILED"
    exit 1
  fi
  echo "[launcher] $(date -Is) [phase=${label}] DONE"
}

phase dataset        --phase dataset --gpu-id 0
phase baselines      --phase baselines --gpu-id 0
phase fp-calibration --phase fp-calibration --gpu-id 0

# Single condition x 3 seeds = 3 cells. Run all 3 seeds in parallel on 3 GPUs
# (one lane per seed; GPU 3 idle this run). shard-id/num-shards=0/1 because
# we pass --condition+--seed explicitly per process (driver fast-path).
COND=on-policy-suppression-cn
wave_all_seeds () {
  echo "[launcher] $(date -Is) [phase=worker-all-seeds] START"
  local pids=() seeds=(42 137 256) gpus=(0 1 2)
  local i
  for i in "${!seeds[@]}"; do
    local seed=${seeds[$i]} gpu=${gpus[$i]}
    # CVD_PIN_EXEMPT: pre-#578 completed-task dispatcher kept verbatim; new launches must pin env CUDA_VISIBLE_DEVICES per cell (gotchas.md CVD-clobber)
    nohup uv run python scripts/run_experiment_444.py \
      --phase worker --shard-id 0 --num-shards 1 \
      --condition "$COND" --seed "$seed" --gpu-id "$gpu" \
      >> "/workspace/logs/issue-444/train_followup_histcn_seed${seed}_${COND}.log" 2>&1 &
    pids+=("$!")
  done
  local fail=0 j=0
  for p in "${pids[@]}"; do
    if ! wait "$p"; then
      echo "[launcher] cell FAILED: ${COND} seed${seeds[$j]} (see train_followup_histcn_seed${seeds[$j]}_${COND}.log)"
      fail=1
    fi
    j=$((j+1))
  done
  if [ "$fail" -ne 0 ]; then
    echo "[launcher] $(date -Is) [phase=worker-all-seeds] FAILED"
    exit 1
  fi
  echo "[launcher] $(date -Is) [phase=worker-all-seeds] DONE"
}
wave_all_seeds

phase full-eval --phase full-eval --gpu-id 0
phase aggregate --phase aggregate
phase upload    --phase upload

echo "[launcher] $(date -Is) [phase=done] FOLLOWUP-histcn COMPLETE"
