#!/usr/bin/env bash
# Launch the issue #170 7-soft + 3-hard sweep on epm-issue-170 (4xH200).
#
# Usage: ssh epm-issue-170 'bash /workspace/explore-persona-space/scripts/launch_issue170_sweep.sh <wave>'
# where <wave> is one of: soft0123, soft456, hard, all-eval
#
# soft0123: launches s0..s3 in parallel on GPUs 0..3 (4-way), ~15.5h wall-time.
# soft456:  launches s4, s5, s6 on GPUs 0..2,             ~15.5h wall-time.
# hard:     launches hardL20, hardL40, hardL80 on GPUs 0..2,  ~6h wall-time.
# all-eval: launches eval for all completed cells, 4-way GPU parallel.
#
# All jobs use nohup; logs go to /workspace/logs/issue-170-<cell>.log
# Each cell uploads its prefix tensor to HF Hub at end of run.

set -euo pipefail

cd /workspace/explore-persona-space

WAVE="${1:-help}"
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"

UV=~/.local/bin/uv

case "$WAVE" in
  soft0123)
    echo "Launching s0..s3 in parallel on GPUs 0..3"
    for i in 0 1 2 3; do
      case "$i" in
        0) cell=s0_K16_lr5e-4 ;;
        1) cell=s1_K32_lr5e-4 ;;
        2) cell=s2_K32_lr1e-4 ;;
        3) cell=s3_K64_lr5e-4 ;;
      esac
      log="$LOGDIR/issue-170-$cell.log"
      echo "  GPU $i -> $cell -> $log"
      nohup env CUDA_VISIBLE_DEVICES=$i $UV run python scripts/run_soft_prefix.py \
        --config-name=$cell > "$log" 2>&1 < /dev/null &
      echo "    PID=$!"
    done
    echo "All 4 launched. Wait with: wait"
    ;;

  soft456)
    echo "Launching s4..s6 on GPUs 0..2"
    cells=(s4_K64_lr1e-4 s5_K64_lr1e-3 s6_K64_lr5e-4_evil_init)
    for i in 0 1 2; do
      cell="${cells[$i]}"
      log="$LOGDIR/issue-170-$cell.log"
      echo "  GPU $i -> $cell -> $log"
      nohup env CUDA_VISIBLE_DEVICES=$i $UV run python scripts/run_soft_prefix.py \
        --config-name=$cell > "$log" 2>&1 < /dev/null &
      echo "    PID=$!"
    done
    echo "3 cells launched"
    ;;

  hard)
    echo "Launching hardL20, hardL40, hardL80 on GPUs 0..2"
    cells=(hardL20 hardL40 hardL80)
    for i in 0 1 2; do
      cell="${cells[$i]}"
      log="$LOGDIR/issue-170-$cell.log"
      echo "  GPU $i -> $cell -> $log"
      nohup env CUDA_VISIBLE_DEVICES=$i $UV run python scripts/run_system_slot_gcg.py \
        --config-name=$cell > "$log" 2>&1 < /dev/null &
      echo "    PID=$!"
    done
    echo "3 hard cells launched"
    ;;

  all-eval)
    echo "Launching eval for all 11 cells, 4-way GPU parallel"
    # Order: pilot (sanity), all 7 soft, all 3 hard, gcg_sanity. = 12 cells.
    declare -a SOFT_CELLS=(
      pilot
      s0_K16_lr5e-4 s1_K32_lr5e-4 s2_K32_lr1e-4 s3_K64_lr5e-4
      s4_K64_lr1e-4 s5_K64_lr1e-3 s6_K64_lr5e-4_evil_init
    )
    declare -a HARD_CELLS=(hardL20 hardL40 hardL80)
    declare -a GCG_CELLS=(gcg_sanity)

    i=0
    for cell in "${SOFT_CELLS[@]}"; do
      gpu=$((i % 4))
      log="$LOGDIR/issue-170-eval-$cell.log"
      echo "  GPU $gpu -> eval $cell (soft)"
      nohup env CUDA_VISIBLE_DEVICES=$gpu $UV run python scripts/eval_issue170_cell.py \
        --cell=$cell --cell-type=soft > "$log" 2>&1 < /dev/null &
      i=$((i + 1))
    done
    for cell in "${HARD_CELLS[@]}"; do
      gpu=$((i % 4))
      log="$LOGDIR/issue-170-eval-$cell.log"
      echo "  GPU $gpu -> eval $cell (hard)"
      nohup env CUDA_VISIBLE_DEVICES=$gpu $UV run python scripts/eval_issue170_cell.py \
        --cell=$cell --cell-type=hard > "$log" 2>&1 < /dev/null &
      i=$((i + 1))
    done
    for cell in "${GCG_CELLS[@]}"; do
      gpu=$((i % 4))
      log="$LOGDIR/issue-170-eval-$cell.log"
      echo "  GPU $gpu -> eval $cell (gcg_sanity)"
      nohup env CUDA_VISIBLE_DEVICES=$gpu $UV run python scripts/eval_issue170_cell.py \
        --cell=$cell --cell-type=gcg_sanity > "$log" 2>&1 < /dev/null &
      i=$((i + 1))
    done
    echo "Launched $i eval jobs"
    ;;

  help|*)
    echo "Usage: $0 <wave>"
    echo "  wave = soft0123 | soft456 | hard | all-eval"
    exit 1
    ;;
esac
