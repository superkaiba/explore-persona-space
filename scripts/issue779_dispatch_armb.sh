#!/usr/bin/env bash
# Issue #779 (training-source-ablation-hg): Arm B/C corpus-gen pod dispatcher.
#
# The ONLY GPU phase of the amendment (plan v6 §4.2/§9). Runs
# issue779_gen_behavior_corpus.py to (1) generate the diverse behavior corpus
# + capture c_last/v(x) + judge, and (2) regenerate + judge the Arm A LMSYS
# g labels (the cached pass_b bundle has no rollout text -> labels must be
# regenerated; a persisted-concern deviation, arm-a-g-labels-require-regen).
# The 2D scaling grid + all free-tier reads run OFF-pod (0-GPU) after upload.
#
# Backend contract (GCP GCE lane OR RunPod): REPO_ROOT defaults to
# $WORKLOAD_ROOT (the GCP lane mirrors /workspace there; RunPod uses /workspace);
# the python driver writes /workspace/logs/issue-779-*.json sentinels
# (poll_pipeline.py-conformant, _SENTINEL_REQUIRED_KEYS) and [phase=...] log
# lines terminating in [phase=done]. Pod-side code NEVER shells out to task.py.
#
# GPU width: the corpus gen is embarrassingly parallel over contexts. On a
# multi-GPU pod, run one trait per GPU (trait-sharded workers), each pinned to
# its own physical GPU via CUDA_VISIBLE_DEVICES + the matching --gpu-id (the
# in-process CVD clobber in the driver is defeated by any import-time cuInit, so
# the LAUNCHER env pin is load-bearing). On 1 GPU, all traits run sequentially.
#
# Usage:
#   bash scripts/issue779_dispatch_armb.sh                 # full run (auto GPU count)
#   bash scripts/issue779_dispatch_armb.sh --smoke         # tiny slice (smoke = sweep)
#   bash scripts/issue779_dispatch_armb.sh --stage corpus  # corpus only (skip g-labels)

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Hub upload acceleration (CLAUDE.md Upload Policy).
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# GCP GCE lane mirrors the repo under $WORKLOAD_ROOT; RunPod uses /workspace.
REPO_ROOT="${WORKLOAD_ROOT:-/workspace/explore-persona-space}"
if [ ! -d "$REPO_ROOT/scripts" ]; then
    # Local/VM smoke fallback: resolve the repo from this script's location.
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

SMOKE=0
STAGE="all"
TRAITS=(evil sycophancy hallucination)
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) SMOKE=1 ;;
        --stage) STAGE="$2"; shift ;;
        --traits) shift; TRAITS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do TRAITS+=("$1"); shift; done; continue ;;
        *) EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

LOG_DIR="$REPO_ROOT/logs/issue_779"
mkdir -p "$LOG_DIR"
mkdir -p /workspace/logs 2>/dev/null || true

echo "[phase=preflight] === issue779 corpus-gen dispatcher $(date -Iseconds) smoke=$SMOKE stage=$STAGE traits=${TRAITS[*]} repo=$REPO_ROOT ==="

# Detect visible GPU count (wave width == detected GPU count; never a hardcoded
# constant — feedback_dispatcher_wave_size_must_match_visible_gpus).
N_GPU=$(uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "[phase=preflight] detected $N_GPU CUDA device(s)"

SMOKE_FLAG=""
[ "$SMOKE" -eq 1 ] && SMOKE_FLAG="--smoke"

# Every worker's stdout/stderr (which includes the DRIVER's own per-run
# [phase=done] terminal line + sentinel echoes) is redirected to a per-worker
# log file, NEVER the main dispatcher log — the [phase=done] token in the MAIN
# log is RESERVED for this dispatcher's single terminal line (incident #545:
# a per-cell [phase=done] echo in the main log produces a false status=done).
run_worker() {
    # $1 = physical gpu id (or "cpu"); $2 = worker label; $3.. = traits
    local gpu="$1"; shift
    local label="$1"; shift
    local wtraits=("$@")
    local wlog="$LOG_DIR/corpus_${label}.log"
    if [ "$gpu" = "cpu" ]; then
        echo "[phase=corpus] CPU worker traits=${wtraits[*]} -> $wlog"
        uv run python scripts/issue779_gen_behavior_corpus.py \
            --stage "$STAGE" --traits "${wtraits[@]}" --device cpu $SMOKE_FLAG \
            "${EXTRA_ARGS[@]}" > "$wlog" 2>&1
    else
        echo "[phase=corpus] GPU $gpu worker traits=${wtraits[*]} -> $wlog"
        # CVD pin in the LAUNCHER env + matching --gpu-id (both required; the
        # in-process clobber is defeated by import-time cuInit).  This IS the
        # pinned shape (no CVD_PIN_EXEMPT needed).
        CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/issue779_gen_behavior_corpus.py \
            --stage "$STAGE" --traits "${wtraits[@]}" --gpu-id "$gpu" $SMOKE_FLAG \
            "${EXTRA_ARGS[@]}" > "$wlog" 2>&1
    fi
}

if [ "$N_GPU" -le 0 ]; then
    # No GPU: single CPU worker (smoke path).
    run_worker cpu cpu "${TRAITS[@]}"
elif [ "$N_GPU" -eq 1 ] || [ "$SMOKE" -eq 1 ]; then
    # 1 GPU (or smoke): all traits sequentially on GPU 0. The driver's per-trait
    # checkpointing + phase-internal upload keep this safe. (Smoke caps traits
    # to 1 inside the driver.)
    run_worker 0 gpu0_all "${TRAITS[@]}"
else
    # Multi-GPU: trait-sharded, one trait per GPU wave, wave width == N_GPU.
    echo "[phase=corpus] multi-GPU trait-sharded: ${#TRAITS[@]} traits over $N_GPU GPU(s)"
    pids=()
    gpu=0
    for t in "${TRAITS[@]}"; do
        run_worker "$gpu" "${t}_gpu${gpu}" "$t" &
        pids+=($!)
        gpu=$(( (gpu + 1) % N_GPU ))
        # Throttle: at most N_GPU concurrent workers.
        if [ "${#pids[@]}" -ge "$N_GPU" ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done
    # Reap remaining.
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[phase=corpus] all trait workers complete"
fi

echo "[phase=done] issue779 corpus-gen dispatcher complete $(date -Iseconds)"
