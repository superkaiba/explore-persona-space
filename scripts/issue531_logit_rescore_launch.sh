#!/usr/bin/env bash
# Issue #531 follow-up — launch the logit re-scoring pass across N GPUs.
# One worker process per GPU, cells split round-robin via --shard i/N.
# Logs to /workspace/logs/issue531_logit_shard<i>.log; per-cell JSONs land in
# eval_results/issue_478/logit_rescore/ (idempotent — re-run safe).
set -euo pipefail

N_GPUS="${1:-4}"
BATCH_SIZE="${2:-16}"

cd "$(dirname "$0")/.."
mkdir -p /workspace/logs

for i in $(seq 0 $((N_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES="$i" nohup uv run python scripts/issue531_logit_rescore.py \
    --shard "$i/$N_GPUS" --batch-size "$BATCH_SIZE" \
    > "/workspace/logs/issue531_logit_shard$i.log" 2>&1 &
  echo "launched shard $i/$N_GPUS on GPU $i (pid $!)"
done

echo "all $N_GPUS shards launched; tail -f /workspace/logs/issue531_logit_shard0.log"
