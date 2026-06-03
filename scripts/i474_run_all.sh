#!/usr/bin/env bash
# Issue #474 — pod-side end-to-end runner (Phases 0..4 on pod; Phase 5 on VM).
#
# Mirrors scripts/i460_run_all.sh: emits [phase=...] markers that
# poll_pipeline.py keys off, plus a 2-min heartbeat that keeps the main
# log mtime within poll's STALL_SEC threshold during long quiet phases.
#
# Plan v3 §4.10 unification: --smoke runs the SAME dispatcher with the
# minimum slice (2 conds x 1 ckpt) — same code path as the full sweep.
#
# Launch (full sweep):
#   nohup bash scripts/i474_run_all.sh > /workspace/logs/issue-474-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-474-run.pid
#
# Launch (smoke only — 2 conds x 1 ckpt):
#   nohup bash scripts/i474_run_all.sh --smoke > /workspace/logs/issue-474-smoke.log 2>&1 &

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

SMOKE_MODE=0
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE_MODE=1 ;;
        *) ;;
    esac
done

# Heartbeat — keeps poll_pipeline alive during quiet phases.
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

run_phase_script() {
    local tag="$1" script="$2"; shift 2
    local rc
    echo "[phase=${tag}] start $(date -Iseconds)"
    bash "scripts/${script}" "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[phase=${tag}] ok $(date -Iseconds)"
        return 0
    fi
    echo "[phase=failed] ${tag} (exit ${rc}) $(date -Iseconds)"
    return "$rc"
}

run_phase_py() {
    local tag="$1" script="$2"; shift 2
    local rc
    echo "[phase=${tag}] start $(date -Iseconds)"
    uv run python "scripts/${script}" "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[phase=${tag}] ok $(date -Iseconds)"
        return 0
    fi
    echo "[phase=failed] ${tag} (exit ${rc}) $(date -Iseconds)"
    return "$rc"
}

run_phase_py    preflight    i474_phase0_preflight.py    || exit 10
# Phase 1 = #460's frozen R; verbatim (A_pos and A_loc SHARE this R).
run_phase_py    rgen         i460_phase1_generate_R.py   || exit 11

if [ "$SMOKE_MODE" -eq 1 ]; then
    # Unified-path smoke per plan v3 §4.10: 2 conds (A1) x 1 ckpt (ep5).
    # Same dispatcher as the sweep, just --smoke-only and reduced epochs.
    run_phase_script train_smoke i474_phase23_dispatch.sh --smoke-only || exit 12
    run_phase_script crosseval_smoke i474_phase4_dispatch.sh --arms pos,loc --epochs 1 || exit 13
else
    run_phase_script train       i474_phase23_dispatch.sh     || exit 12
    run_phase_script crosseval   i474_phase4_dispatch.sh      || exit 13
fi

# Phase 5 runs on the VM after artifacts sync (it consumes JSON only).
echo "[phase=done] phases 0..4 complete $(date -Iseconds)"
