#!/usr/bin/env bash
# CPU-only existing-artifact analysis. Invoke from the producing worktree.
set -euo pipefail
if [[ "$#" -ne 2 ]]; then
  echo "usage: bash scripts/launch_issue2546_necessity_rank.sh RUN_LABEL ABS_LOG" >&2
  exit 2
fi
RUN_LABEL="$1"
RUN_LOG="$2"
RANK_WT="/home/thomasjiralerspong/wt-cot-rank-qwen3"
RANK_REPO="/home/thomasjiralerspong/explore-persona-space"
case "$RUN_LOG" in "$RANK_WT"/*) ;; *) echo "Log must be inside analysis worktree" >&2; exit 2;; esac
if [[ -e "$RUN_LOG" ]]; then
  echo "Refusing to append to an existing run log: $RUN_LOG" >&2
  exit 2
fi
if pgrep -f 'python.*scripts.issue2546_necessity_ran[k]' >/dev/null; then
  echo "Existing necessity-rank worker detected; refusing duplicate launch" >&2
  exit 3
fi
cd "$RANK_WT"
setsid nohup env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 \
  UV_PROJECT_ENVIRONMENT="$RANK_REPO/.venv" UV_NO_SYNC=1 \
  PYTHONPATH="$RANK_WT:$RANK_WT/src" \
  uv run --no-sync python -u -m scripts.issue2546_necessity_rank \
    --data-root /mnt/eps-data/thomasjiralerspong/cot_necessity \
    --labels "$RANK_REPO/eval_results/issue_2546/necessity/qwen3_toggle_labels.json" \
    --out "$RANK_WT/eval_results/issue_2546/qwen3_necessity_rank" \
    --run-label "$RUN_LABEL" < /dev/null > "$RUN_LOG" 2>&1 &
RANK_LAUNCH_PID=$!
printf 'launcher_pid=%s log=%s harvest=%s\n' "$RANK_LAUNCH_PID" "$RUN_LOG" \
  "$RANK_WT/eval_results/issue_2546/qwen3_necessity_rank"
