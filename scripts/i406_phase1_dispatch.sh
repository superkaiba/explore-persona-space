#!/usr/bin/env bash
# Phase 1 — 2-GPU parallel dispatcher.
#
# Splits the 50 Q_test probes across 2 GPUs by q_idx % 2. Each shard holds
# a full base-model copy on its 80 GB GPU. After both finish, runs the
# merger to assemble D_matrix.json + D_per_position.json + C_L*.json.
#
# Usage:
#     bash scripts/i406_phase1_dispatch.sh
#     # Resume after a partial run:
#     bash scripts/i406_phase1_dispatch.sh --resume
#
# Per-shard stdout/stderr lands in logs/issue_406/phase1_gpu{0,1}.log.

set -euo pipefail

EXTRA_FLAGS="${*:-}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

LOG_DIR=logs/issue_406
mkdir -p "$LOG_DIR"

echo "=== Phase 1 dispatcher starting at $(date -Iseconds) ==="
echo "    Extra flags forwarded to each shard: ${EXTRA_FLAGS:-<none>}"
echo "    Logs: $LOG_DIR/phase1_gpu0.log + phase1_gpu1.log"

# Shard 0 on physical GPU 0
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/i406_phase1_compute_divergence.py \
    --gpu-shard 0-of-2 ${EXTRA_FLAGS} \
    > "$LOG_DIR/phase1_gpu0.log" 2>&1 &
GPU0_PID=$!

# Shard 1 on physical GPU 1
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts/i406_phase1_compute_divergence.py \
    --gpu-shard 1-of-2 ${EXTRA_FLAGS} \
    > "$LOG_DIR/phase1_gpu1.log" 2>&1 &
GPU1_PID=$!

echo "    GPU 0 PID: $GPU0_PID; GPU 1 PID: $GPU1_PID"
wait $GPU0_PID $GPU1_PID

echo "=== Phase 1 shards complete; running merger ==="
uv run python scripts/i406_phase1_merge_and_compute_matrices.py

echo "=== Phase 1 done at $(date -Iseconds) ==="
