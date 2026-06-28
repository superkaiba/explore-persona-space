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
# loaded at the weight level -- so the launcher trains the whole fleet itself.
#
# #664 r16: p1's CVD pin was previously `CUDA_VISIBLE_DEVICES=$shard ... --gpu-id 0`
# on the false assumption (since-corrected in train/sft.py) that the env wins over
# the in-process gpu_id clobber. It does NOT: `_warn_if_cvd_disagrees` in
# src/explore_persona_space/train/sft.py:151 warns then UNCONDITIONALLY rewrites
# `os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)` (line 1221 for train_lora,
# 1583 for merge_lora). With --gpu-id 0 forced, every shard collapsed onto physical
# GPU 0 and all 8 OOMed (observed 2026-06-28). Fix: pass --gpu-id "$shard" so
# train/sft.py pins each shard to its own physical GPU.
#
# Cells are partitioned by `i % num_shards == shard_id` inside run_all, so shard i
# trains AND then extract+evals exactly the cells it owns (train_cell writes the
# adapter dir p2 reads + pushes it to HF). train_cell is idempotent (skips a cell
# whose adapter_model.safetensors already exists), so a restart at --phase p1
# resumes cleanly. The 48 r1-r6 adapters on HF are NOT reused (plan §4); the fresh
# training overwrites them via train_lora's normal upload.
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
#   nohup bash scripts/issue664_launch_parallel.sh p1 < /dev/null &   # resume at p1 (skip p0)
set -euo pipefail
cd /workspace/explore-persona-space

# START_PHASE selects the first phase to run (p0|p1|p2|p3). Default p0 runs
# the full sequence. Used to resume after a transient crash in a later phase
# without redoing the ~25-min p0 setup.
START_PHASE="${1:-p0}"
case "$START_PHASE" in
    p0|p1|p2|p3) ;;
    *) echo "[wrapper] ERROR: START_PHASE must be one of p0|p1|p2|p3, got '$START_PHASE'" >&2; exit 2 ;;
esac
declare -A _PHASE_ORDER=([p0]=0 [p1]=1 [p2]=2 [p3]=3)
_start_n="${_PHASE_ORDER[$START_PHASE]}"
should_run() {
    local n="${_PHASE_ORDER[$1]}"
    [ "$n" -ge "$_start_n" ]
}

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"
echo $$ > "$LOG_DIR/issue664-run.pid"

# Tee everything to the watched top-level log. poll_pipeline.py parses
# `[phase=<name>]` (PHASE_RE) from this file; emit one per phase + a terminal
# `[phase=done]` on graceful completion ONLY.
exec > >(tee -a "$LOG_DIR/issue664-run.log") 2>&1
echo "[wrapper] $(date -Iseconds) starting issue664_launch_parallel.sh pid=$$ START_PHASE=$START_PHASE"

NUM_SHARDS=8

# ── Phase 0 ── single process, GPU 0 (global setup; must run exactly once).
if should_run p0; then
    echo "[phase=p0_setup]"
    echo "[wrapper] phase=p0 (single shard, GPU 0)"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue664_dispatch.py --phase p0 --gpu-id 0 \
        --shard-id 0 --num-shards "$NUM_SHARDS" \
        > "$LOG_DIR/issue664-p0.log" 2>&1  # CVD_PIN_EXEMPT: sequential single-GPU setup, not parallel
    echo "[wrapper] phase=p0 done"
else
    echo "[wrapper] phase=p0 SKIPPED (START_PHASE=$START_PHASE)"
fi

# ── Phase 1 ── 8-way data-parallel TRAIN. Each shard pinned to its own physical GPU
# via --gpu-id "$shard": train/sft.py's _warn_if_cvd_disagrees rewrites
# os.environ["CUDA_VISIBLE_DEVICES"] from cfg.gpu_id (unconditional clobber, both
# train_lora and merge_lora), so the inherited env CVD is ignored and gpu_id is the
# only authoritative pin. Passing --gpu-id "$shard" makes train/sft.py set CVD=$shard
# correctly; the leading `CUDA_VISIBLE_DEVICES=$shard` is kept as belt-and-suspenders
# (it agrees with cfg.gpu_id, so _warn_if_cvd_disagrees stays silent). Shard i trains
# the same cells it extract+evals at p2 (i % num_shards == shard_id partition inside
# run_all). train_cell is idempotent (skip-if-adapter-exists), so a restart at
# --phase p1 resumes without retraining cells whose adapter dirs already landed.
if should_run p1; then
    echo "[phase=p1_train]"
    echo "[wrapper] phase=p1 ($NUM_SHARDS shards in parallel)"
    pids=()
    for shard in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$shard uv run python scripts/issue664_dispatch.py --phase p1 --gpu-id "$shard" --shard-id "$shard" --num-shards "$NUM_SHARDS" > "$LOG_DIR/issue664-p1-shard${shard}.log" 2>&1 &
        pids+=($!)
        echo "[wrapper] launched p1 shard $shard pid=${pids[-1]} (gpu_id=$shard)"
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
else
    echo "[wrapper] phase=p1 SKIPPED (START_PHASE=$START_PHASE)"
fi

# ── Phase 2 ── 8-way data-parallel extract+eval. Same CVD discipline as p1:
# merge_lora (sft.py:1583) also calls _warn_if_cvd_disagrees + rewrites
# CUDA_VISIBLE_DEVICES=str(gpu_id), so --gpu-id "$shard" is the authoritative
# per-shard pin. The leading `CUDA_VISIBLE_DEVICES=$shard` agrees and keeps the
# warning silent.
if should_run p2; then
    echo "[phase=p2_extract_eval]"
    echo "[wrapper] phase=p2 ($NUM_SHARDS shards in parallel)"
    pids=()
    for shard in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$shard uv run python scripts/issue664_dispatch.py --phase p2 --gpu-id "$shard" --shard-id "$shard" --num-shards "$NUM_SHARDS" > "$LOG_DIR/issue664-p2-shard${shard}.log" 2>&1 &
        pids+=($!)
        echo "[wrapper] launched p2 shard $shard pid=${pids[-1]} (gpu_id=$shard)"
    done
    # Wait for all shards; collect failures (don't let `set -e` short-circuit the wait loop).
    fail=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[wrapper] OK:     p2 shard $i pid=${pids[$i]}"
        else
            echo "[wrapper] FAILED: p2 shard $i pid=${pids[$i]}"
            fail=$((fail + 1))
        fi
    done
    if [ "$fail" -gt 0 ]; then
        # rc=N in the watched log makes poll_pipeline read NOT-done even if a stray
        # done token appears later (PHASE_DONE_NEGATION_RE matches `rc=[1-9]`).
        echo "[wrapper] $fail p2 shard(s) failed rc=1; aborting before p3 -- [phase=done] NOT emitted"
        exit 1
    fi
    echo "[wrapper] phase=p2 done (all $NUM_SHARDS shards OK)"
else
    echo "[wrapper] phase=p2 SKIPPED (START_PHASE=$START_PHASE)"
fi

# ── Phase 3 ── single process, GPU 0. Runs AFTER `wait` for every p2 shard, so the
# full marker_slot_stats.json set exists for the A7 readability assert (#664 r8). p3 =
# all-fleet manifest + A7 readability assert + raw-completions upload + store-tensor
# upload + propensity covariate upload, then the dispatcher writes the epm:results
# sentinel.
if should_run p3; then
    echo "[phase=p3_upload]"
    echo "[wrapper] phase=p3 (single shard, GPU 0)"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue664_dispatch.py --phase p3 --gpu-id 0 \
        --shard-id 0 --num-shards "$NUM_SHARDS" \
        > "$LOG_DIR/issue664-p3.log" 2>&1  # CVD_PIN_EXEMPT: sequential single-GPU upload, not parallel
    echo "[wrapper] phase=p3 done"
else
    echo "[wrapper] phase=p3 SKIPPED (START_PHASE=$START_PHASE)"
fi

echo "[wrapper] $(date -Iseconds) issue664_launch_parallel.sh COMPLETE"
# RESERVED terminal line: the dispatcher's phased runs do NOT emit [phase=done];
# this wrapper emits the single watched terminal token after p3 (sentinel) succeeds.
echo "[phase=done]"
