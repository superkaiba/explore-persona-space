#!/usr/bin/env bash
# scripts/issue664_launch_parallel.sh
# Wrapper to orchestrate issue664_dispatch.py for 8-way DATA parallelism on an
# 8xH100 pod (#664 round-7). The per-cell science behavior is unchanged; this only
# fans the independent (source x behavior x arm x dose) cells across 8 GPUs.
#
#   1. p0 once          (shard 0, single GPU 0)  -- global setup: caches/pools/mixes
#   2. p2 8-way parallel (8 shards, each pinned to its own physical GPU)
#   3. p3 once          (shard 0, single GPU 0)  -- manifest/asserts/upload + sentinel
#
# Cells are partitioned by `i % num_shards == shard_id` AFTER the post-P2.0 drop
# filter, inside issue664_dispatch.run_all. The shard-0-only phases (manifest,
# readability assert, upload, results sentinel) operate over the FULL fleet.
#
# Pod-side contract (CLAUDE.md / poll_pipeline.py): this top-level teed log carries
# the watched `[phase=<name>]` lines and the SINGLE terminal `[phase=done]` (the
# per-phase logs the dispatcher writes are NOT the watched surface). The dispatcher's
# `--phase p3` shard-0 run writes the end-of-run epm:results sentinel JSON.
#
# Usage (on the pod):
#   nohup bash scripts/issue664_launch_parallel.sh < /dev/null &
set -euo pipefail
cd /workspace/explore-persona-space

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"
echo $$ > "$LOG_DIR/issue664-run.pid"

# Tee everything to the watched top-level log. poll_pipeline.py parses
# `[phase=<name>]` (PHASE_RE) from this file; emit one per phase + a terminal
# `[phase=done]` on graceful completion ONLY.
exec > >(tee -a "$LOG_DIR/issue664-run.log") 2>&1
echo "[wrapper] $(date -Iseconds) starting issue664_launch_parallel.sh pid=$$"

NUM_SHARDS=8

# ── Phase 0 ── single process, GPU 0 (global setup; must run exactly once).
echo "[phase=p0_setup]"
echo "[wrapper] phase=p0 (single shard, GPU 0)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue664_dispatch.py --phase p0 --gpu-id 0 \
    --shard-id 0 --num-shards "$NUM_SHARDS" \
    > "$LOG_DIR/issue664-p0.log" 2>&1  # CVD_PIN_EXEMPT: sequential single-GPU setup, not parallel
echo "[wrapper] phase=p0 done"

# ── Phase 2 ── 8-way data-parallel. Each shard pinned to its own physical GPU via
# CUDA_VISIBLE_DEVICES=$shard; --gpu-id 0 is correct because CVD remaps the visible
# device to index 0 inside the process (and train/sft.py's in-process CVD clobber
# rewrites the already-constrained single-GPU view to "0").
echo "[phase=p2_extract_eval]"
echo "[wrapper] phase=p2 ($NUM_SHARDS shards in parallel)"
pids=()
for shard in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$shard uv run python scripts/issue664_dispatch.py --phase p2 --gpu-id 0 --shard-id "$shard" --num-shards "$NUM_SHARDS" > "$LOG_DIR/issue664-p2-shard${shard}.log" 2>&1 &
    pids+=($!)
    echo "[wrapper] launched shard $shard pid=${pids[-1]}"
done
# Wait for all shards; collect failures (don't let `set -e` short-circuit the wait loop).
fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[wrapper] OK:     shard $i pid=${pids[$i]}"
    else
        echo "[wrapper] FAILED: shard $i pid=${pids[$i]}"
        fail=$((fail + 1))
    fi
done
if [ "$fail" -gt 0 ]; then
    # rc=N in the watched log makes poll_pipeline read NOT-done even if a stray
    # done token appears later (PHASE_DONE_NEGATION_RE matches `rc=[1-9]`).
    echo "[wrapper] $fail shard(s) failed rc=1; aborting before p3 -- [phase=done] NOT emitted"
    exit 1
fi
echo "[wrapper] phase=p2 done (all $NUM_SHARDS shards OK)"

# ── Phase 3 ── single process, GPU 0. Manifest + readability assert + upload over
# the full fleet, then the dispatcher writes the epm:results sentinel.
echo "[phase=p3_upload]"
echo "[wrapper] phase=p3 (single shard, GPU 0)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue664_dispatch.py --phase p3 --gpu-id 0 \
    --shard-id 0 --num-shards "$NUM_SHARDS" \
    > "$LOG_DIR/issue664-p3.log" 2>&1  # CVD_PIN_EXEMPT: sequential single-GPU upload, not parallel
echo "[wrapper] phase=p3 done"

echo "[wrapper] $(date -Iseconds) issue664_launch_parallel.sh COMPLETE"
# RESERVED terminal line: the dispatcher's phased runs do NOT emit [phase=done];
# this wrapper emits the single watched terminal token after p3 (sentinel) succeeds.
echo "[phase=done]"
