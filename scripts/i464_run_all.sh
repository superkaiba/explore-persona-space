#!/usr/bin/env bash
# Issue #464 — pod-side end-to-end runner (Phases 0..5 on pod).
#
# Mirrors scripts/i460_run_all.sh: emits [phase=...] markers that
# poll_pipeline.py keys off, plus a 2-min heartbeat keeping the main
# log mtime fresh during long quiet phases.
#
# Phase tags use digit-free names (PHASE_RE = \[phase=([a-z_]+)).
#
# Launch:
#   nohup bash scripts/i464_run_all.sh > /workspace/logs/issue-464-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-464-run.pid

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

# Heartbeat (CLAUDE.md / #460 mirror).
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

# Round-2 #460 fix style: capture rc IMMEDIATELY after the command.
run_phase_script() {
    local tag="$1" script="$2" rc
    echo "[phase=${tag}] start $(date -Iseconds)"
    bash "scripts/${script}"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[phase=${tag}] ok $(date -Iseconds)"
        return 0
    fi
    echo "[phase=failed] ${tag} (exit ${rc}) $(date -Iseconds)"
    return "$rc"
}

run_phase_py() {
    local tag="$1" script="$2" rc
    shift 2 || true
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

run_phase_py     preflight       i464_phase0_preflight.py        || exit 10
run_phase_py     rgen            i464_phase1_generate_R.py       || exit 11
run_phase_script train           i464_phase23_dispatch.sh        || exit 12
run_phase_script crosseval       i464_phase4_dispatch.sh         || exit 13
run_phase_py     onpolicy        i464_phase45_onpolicy_validation.py || exit 14
run_phase_py     analyze         i464_phase5_analyze.py          || exit 15
run_phase_py     plot            plot_i464_clean_result.py       || exit 16

# Final marker required by poll_pipeline.py for "done".
echo "[phase=done] phases 0..5 complete $(date -Iseconds)"
