#!/usr/bin/env bash
# scripts/issue664_launch_parallel.sh
# Wrapper to orchestrate issue664_dispatch.py for 8-way DATA parallelism on an
# 8xH100 pod (#664 round-7). The per-cell science behavior is unchanged; this only
# fans the independent (source x behavior x arm x dose) cells across 8 GPUs.
#
#   1. p0 once          (shard 0, single GPU 0)  -- global setup: caches/pools/mixes
#   2. p1 8-way parallel (8 shards, each pinned to its own physical GPU)  -- TRAIN
#   3. p2 8-way parallel (8 shards, each pinned to its own physical GPU)  -- extract+eval
#   4. p3 once          (shard 0, single GPU 0)  -- manifest/asserts/upload + sentinel
#
# #664 r15: p1 (training) was previously SKIPPED here on the assumption that an
# EXTERNAL step trained the fleet; on a fresh pod with no local-volume persistence
# that assumption breaks silently -- p2 finds no local adapter to merge and crashes
# (concern p2-no-adapter-rehydrate-train-skipped). The plan (§4 "Recipe-only reuse
# (HARD constraint)") requires training EVERY cell fresh -- no prior adapter is
# loaded at the weight level -- so the launcher trains the whole fleet itself. p1
# uses the identical 8-way CVD-pinned fan-out as p2; cells are partitioned by
# `i % num_shards == shard_id` inside run_all, so shard i trains AND then
# extract+evals exactly the cells it owns (train_cell writes the adapter dir p2
# reads + pushes it to HF). train_cell is idempotent (skips a cell whose
# adapter_model.safetensors already exists), so a restart at --phase p1 resumes
# cleanly. The 48 r1-r6 adapters on HF are NOT reused (plan §4); the fresh training
# overwrites them via train_lora's normal upload.
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

# ── Phase 1 ── 8-way data-parallel TRAIN. Each shard pinned to its own physical GPU
# via CUDA_VISIBLE_DEVICES=$shard (the in-process clobber in train/sft.py is silently
# defeated by import-time cuInit, so the launcher-env pin is load-bearing -- gotchas
# §"in-process CVD clobber is defeated by import-time cuInit"). --gpu-id 0 is correct
# because CVD remaps the visible device to index 0 inside the process. Shard i trains
# the same cells it extract+evals at p2 (i % num_shards == shard_id partition inside
# run_all), so each adapter dir exists locally before this shard's p2 reads it; it is
# also pushed to HF by train_lora. train_cell is idempotent (skip-if-adapter-exists),
# so a restart resumes without retraining. p1 must finish (wait) before p2: p2 merges
# each shard's adapter, which only exists after that shard's p1 train_cell wrote it.
echo "[phase=p1_train]"
echo "[wrapper] phase=p1 ($NUM_SHARDS shards in parallel)"
pids=()
for shard in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$shard uv run python scripts/issue664_dispatch.py --phase p1 --gpu-id 0 --shard-id "$shard" --num-shards "$NUM_SHARDS" > "$LOG_DIR/issue664-p1-shard${shard}.log" 2>&1 &
    pids+=($!)
    echo "[wrapper] launched p1 shard $shard pid=${pids[-1]}"
done
# Wait for all p1 shards; collect failures (don't let `set -e` short-circuit the loop).
fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[wrapper] OK:     p1 shard $i pid=${pids[$i]}"
    else
        echo "[wrapper] FAILED: p1 shard $i pid=${pids[$i]}"
        fail=$((fail + 1))
    fi
done
if [ "$fail" -gt 0 ]; then
    # rc=N in the watched log makes poll_pipeline read NOT-done even if a stray
    # done token appears later (PHASE_DONE_NEGATION_RE matches `rc=[1-9]`).
    echo "[wrapper] $fail p1 shard(s) failed rc=1; aborting before p2 -- [phase=done] NOT emitted"
    exit 1
fi
echo "[wrapper] phase=p1 done (all $NUM_SHARDS shards OK)"

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

# ── Phase 3 ── single process, GPU 0. Runs AFTER `wait` for every p2 shard, so the
# full marker_slot_stats.json set exists for the A7 readability assert (#664 r8). p3 =
# all-fleet manifest + A7 readability assert + raw-completions upload + store-tensor
# upload + propensity covariate upload, then the dispatcher writes the epm:results
# sentinel.
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
