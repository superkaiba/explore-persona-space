#!/usr/bin/env bash
# Issue #1689 follow-up round `real-u2-capture` — five-phase dispatcher.
#
# Phase A0: filter LMSYS + WildChat multi-turn corpus, sample ~3800 convs
# Phase A1: generate Haiku companion u2 for the SAME 3800 (u1, a1) pairs
# Phase A2+B: render + teacher-forced L19 capture (12 cells)
# Phase C: fits battery (prefix + context arms, identity+bias + kNN + null)
# Phase D: analyzer fold-back (0 GPU, not driven from this script)
#
# Smoke: SMOKE=1 (env) narrows every phase's slice:
#   A0: keep_target=20, max_scan=5000
#   A1: 5 rows + mock response (no API call)
#   A2+B: 20 rows, 12 cells, allow-cpu
#   C: null_draws=5

set -euo pipefail

# --- Env resolution ---------------------------------------------------------

# REPO_ROOT: on the GCE lane the startup script exports WORKLOAD_ROOT; on
# RunPod / local, resolve from script location.
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"

# Conditional .env sourcing — GCE has no .env file, its startup exports
# tokens directly (gotchas.md § conditional sourcing for GCP/pod scripts).
if [ -f ./.env ]; then
    set -a
    . ./.env
    set +a
fi

# Shared-VM CPU thread caps (code-style.md § shared-VM thread caps).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

SMOKE="${SMOKE:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/issue_1689_real_u2_capture}"
mkdir -p "$LOG_DIR"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/issue_1689/real_u2_capture}"
CORPUS_PATH="${DATA_ROOT}/corpus/real_multiturn_first_exchange.jsonl"
HAIKU_PATH="${DATA_ROOT}/raw_completions/haiku_u2.jsonl"
STORE_ROOT="${DATA_ROOT}/store"
EVAL_ROOT="${REPO_ROOT}/eval_results/issue_1689/real_u2_capture"

# Smoke flag conversion
SMOKE_FLAG=""
if [ "$SMOKE" = "1" ]; then
    SMOKE_FLAG="--smoke"
fi

# --- Phase dispatch ---------------------------------------------------------

run_phase_a0() {
    echo "[phase=a0_corpus] starting SMOKE=$SMOKE" >&2
    uv run python "${REPO_ROOT}/scripts/issue1689_real_u2_gen_corpus.py" \
        --out "$CORPUS_PATH" \
        $SMOKE_FLAG \
        2>&1 | tee "${LOG_DIR}/phase_a0.log"
    echo "[phase=a0_corpus] done" >&2
}

run_phase_a1() {
    echo "[phase=a1_haiku] starting SMOKE=$SMOKE" >&2
    uv run python "${REPO_ROOT}/scripts/issue1689_real_u2_haiku_gen.py" \
        --in "$CORPUS_PATH" \
        --out "$HAIKU_PATH" \
        $SMOKE_FLAG \
        2>&1 | tee "${LOG_DIR}/phase_a1.log"
    echo "[phase=a1_haiku] done" >&2
}

run_phase_capture() {
    echo "[phase=capture] starting SMOKE=$SMOKE" >&2
    local device_flag=""
    if [ "$SMOKE" = "1" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
        device_flag="--device cpu"
    fi
    uv run python "${REPO_ROOT}/scripts/issue1689_real_u2_capture.py" \
        --corpus "$CORPUS_PATH" \
        --haiku "$HAIKU_PATH" \
        --out-root "$STORE_ROOT" \
        $device_flag \
        $SMOKE_FLAG \
        2>&1 | tee "${LOG_DIR}/phase_capture.log"
    echo "[phase=capture] done" >&2
}

run_phase_fits() {
    echo "[phase=fits] starting SMOKE=$SMOKE" >&2
    uv run python "${REPO_ROOT}/scripts/issue1689_real_u2_fits.py" \
        --store-root "$STORE_ROOT" \
        --out-dir "$EVAL_ROOT" \
        $SMOKE_FLAG \
        2>&1 | tee "${LOG_DIR}/phase_fits.log"
    echo "[phase=fits] done" >&2
}

run_phase_upload() {
    # Round-2 Major #1: wire HF upload BEFORE [phase=done], so Step-8
    # upload-verifier PASSes without a fix round.
    #
    # Uploads the whole data/issue_1689/real_u2_capture/ tree to
    # superkaiba1/explore-persona-space-data/issue1689_speaker_lattice/real_u2_capture/
    # in ONE upload_folder commit, wrapped in hub.retry_transient for the
    # shared HF-fleet rate limit. Fits eval JSONs are committed to git on
    # the issue branch by the workload (Step-8 syncs them separately).
    echo "[phase=upload] starting SMOKE=$SMOKE" >&2
    uv run python "${REPO_ROOT}/scripts/issue1689_real_u2_upload.py" \
        --data-root "$DATA_ROOT" \
        $SMOKE_FLAG \
        2>&1 | tee "${LOG_DIR}/phase_upload.log"
    echo "[phase=upload] done" >&2
}

run_all() {
    run_phase_a0
    run_phase_a1
    run_phase_capture
    run_phase_fits
    run_phase_upload
    echo "[phase=done]" >&2
}

# --- Entry point ------------------------------------------------------------

PHASE="${1:-all}"
case "$PHASE" in
    a0|corpus) run_phase_a0 ;;
    a1|haiku) run_phase_a1 ;;
    capture) run_phase_capture ;;
    fits) run_phase_fits ;;
    upload) run_phase_upload ;;
    all|run_phase_real_u2_capture) run_all ;;
    *)
        echo "usage: $0 {all|a0|a1|capture|fits|upload}" >&2
        exit 2
        ;;
esac
