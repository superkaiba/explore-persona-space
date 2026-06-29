#!/usr/bin/env bash
# Issue 664 r23 recovery wrapper (Path 2 — cherry-picked HF-aware-skip primitives).
#
# State at launch (verified r23): pod has ONLY the 20 mk_* cells local; the 48
# non-rf/sy cells are FULLY on HF (raw completions + store tensors); the 16 rf/sy
# cells (sycophancy + refusal × {default,librarian} × {contra,posonly} × {d1,d2},
# seed 42) were NEVER trained (absent everywhere). The cherry-picked
# _cell_done_anywhere / _cell_artifacts_on_hub primitives make p2 SKIP every cell
# already complete locally OR on HF, and the p3 finalizers SKIP HF-complete cells
# (upload only the fresh ones) — so this wrapper only generates + uploads the 16
# rf/sy cells, then p3 finalizes the full 64-cell fleet idempotently.
#
# Recipe (plan §11):
#   sycophancy: lr 1e-5, 3 epochs            (Source: #537/#411)
#   refusal:    lr 1e-4, r16/α32, 3 epochs   (Source: #537/#390)
#   (lr/epochs/rank are resolved inside the dispatcher's per-behavior train_cell;
#    this wrapper only sequences the phases + sets the shared env.)
#
# EPM_MARKER_READ_GAUGE=optB stays on for consistency with r22 (marker-only; a
# harmless no-op for the rf/sy content behaviors here).
#
# Per-behavior p0/p1/p2 run sequentially (sycophancy then refusal); within each
# behavior the p1 (train) + p2 (extract+eval) loops fan across 4 shards / 4 GPUs
# (CUDA_VISIBLE_DEVICES per shard), matching r22's 4-way data parallel. p0 is
# shard-0-only global setup (builds the behavior's training mixes + on-policy
# caches; idempotent on the restored baseline cache). p3 is full-fleet, single
# process, online (HF push) — the cherry-picked finalizers skip the 48+20 on HF.
#
# Pod-side: writes [phase=...] log lines + the dispatcher's own epm:results
# sentinel (via _write_results_sentinel at p3 shard-0). NEVER shells task.py.
set -euo pipefail
cd /workspace/explore-persona-space

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"
echo $$ > "$LOG_DIR/issue664-r23-run.pid"
exec > >(tee -a "$LOG_DIR/issue664-r23-run.log") 2>&1
echo "[wrapper-r23] $(date -Iseconds) starting Path-2 recovery pid=$$ (4 shards x 1 GPU)"

# load credentials (HF_TOKEN / WANDB_API_KEY) for the dispatcher subprocesses.
set -a
# shellcheck disable=SC1091
[ -f .env ] && . ./.env
set +a

NUM_SHARDS=4

run_behavior() {
    local behavior="$1"
    echo "[phase=p0_${behavior}]"
    echo "[wrapper-r23] phase=p0 behavior=${behavior} (shard-0 global setup: build mixes + caches)"
    # p0 is shard-0-only global setup; one process. Offline mode is NOT set here —
    # p0's sycophancy/refusal on-policy elicitation needs the base model from cache
    # (already present) but also reads the HF data repo for pools; leave online.
    EPM_MARKER_READ_GAUGE=optB \
        uv run python scripts/issue664_dispatch.py \
        --phase p0 --gpu-id 0 --shard-id 0 --num-shards "$NUM_SHARDS" \
        --marker-read-gauge optB --behavior "$behavior" \
        >"$LOG_DIR/issue664-r23-p0-${behavior}.log" 2>&1
    echo "[wrapper-r23] phase=p0 behavior=${behavior} done"

    echo "[phase=p1_${behavior}]"
    echo "[wrapper-r23] phase=p1 behavior=${behavior} (${NUM_SHARDS} shards train)"
    local pids=()
    for shard in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$shard \
        EPM_MARKER_READ_GAUGE=optB \
            uv run python scripts/issue664_dispatch.py \
            --phase p1 --gpu-id "$shard" --shard-id "$shard" --num-shards "$NUM_SHARDS" \
            --marker-read-gauge optB --behavior "$behavior" \
            >"$LOG_DIR/issue664-r23-p1-${behavior}-shard${shard}.log" 2>&1 &
        pids+=($!)
        echo "[wrapper-r23] launched p1 ${behavior} shard $shard pid=${pids[-1]} (gpu $shard)"
    done
    local fail=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[wrapper-r23] OK:     p1 ${behavior} shard $i pid=${pids[$i]}"
        else
            echo "[wrapper-r23] FAILED: p1 ${behavior} shard $i pid=${pids[$i]} rc=$?"
            fail=$((fail + 1))
        fi
    done
    if [ "$fail" -gt 0 ]; then
        echo "[wrapper-r23] $fail p1 ${behavior} shard(s) failed; aborting before p2 -- [phase=done] NOT emitted"
        exit 1
    fi
    echo "[wrapper-r23] phase=p1 behavior=${behavior} done (all $NUM_SHARDS shards OK)"

    echo "[phase=p2_${behavior}]"
    echo "[wrapper-r23] phase=p2 behavior=${behavior} (${NUM_SHARDS} shards extract+eval)"
    pids=()
    for shard in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$shard \
        EPM_MARKER_READ_GAUGE=optB \
            uv run python scripts/issue664_dispatch.py \
            --phase p2 --gpu-id "$shard" --shard-id "$shard" --num-shards "$NUM_SHARDS" \
            --marker-read-gauge optB --behavior "$behavior" \
            >"$LOG_DIR/issue664-r23-p2-${behavior}-shard${shard}.log" 2>&1 &
        pids+=($!)
        echo "[wrapper-r23] launched p2 ${behavior} shard $shard pid=${pids[-1]} (gpu $shard)"
    done
    fail=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[wrapper-r23] OK:     p2 ${behavior} shard $i pid=${pids[$i]}"
        else
            echo "[wrapper-r23] FAILED: p2 ${behavior} shard $i pid=${pids[$i]} rc=$?"
            fail=$((fail + 1))
        fi
    done
    if [ "$fail" -gt 0 ]; then
        echo "[wrapper-r23] $fail p2 ${behavior} shard(s) failed; aborting before next behavior -- [phase=done] NOT emitted"
        exit 1
    fi
    echo "[wrapper-r23] phase=p2 behavior=${behavior} done (all $NUM_SHARDS shards OK)"
}

# The 16 fresh cells: sycophancy then refusal (sequential, each 4-shard parallel).
run_behavior sycophancy
run_behavior refusal

# p3 full-fleet finalize (NO --behavior filter -- p3 describes/uploads the WHOLE
# 64-cell fleet). The cherry-picked HF-aware finalizers SKIP the 48 + 20 mk_*
# cells already on HF and upload only the 16 fresh rf/sy cells; the A7 readability
# assert reads the 20 mk_* marker_slot stats locally (they PASSed at r22 13:38).
# Online (HF push). Single process, shard 0 -> writes the epm:results sentinel +
# the [phase=done] terminal line.
echo "[phase=p3_upload]"
echo "[wrapper-r23] phase=p3 full-fleet finalize (HF-aware skip; uploads only the 16 fresh cells)"
EPM_MARKER_READ_GAUGE=optB \
    uv run python scripts/issue664_dispatch.py \
    --phase p3 --gpu-id 0 --shard-id 0 --num-shards "$NUM_SHARDS" \
    --marker-read-gauge optB \
    >"$LOG_DIR/issue664-r23-p3.log" 2>&1
echo "[wrapper-r23] phase=p3 done"

echo "[phase=done]"
echo "[wrapper-r23] $(date -Iseconds) r23 Path-2 recovery complete"
