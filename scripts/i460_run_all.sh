#!/usr/bin/env bash
# Issue #460 — pod-side end-to-end runner (Phases 0..4 on pod; Phase 5 on VM).
#
# Mirrors scripts/i406_run_all.sh: emits [phase=...] markers that
# poll_pipeline.py keys off, plus a 2-min heartbeat that keeps the main
# log mtime within poll's STALL_SEC threshold during long quiet phases.
#
# Phase tags use digit-free names (PHASE_RE = \[phase=([a-z_]+)).
#
# Launch:
#   nohup bash scripts/i460_run_all.sh > /workspace/logs/issue-460-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-460-run.pid

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

# Heartbeat: keep main-log mtime fresh during quiet single-condition trains.
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

# Round-2 fix (#460, 2026-06-01): capture $? IMMEDIATELY after the command,
# NOT inside the `if !` branch. The previous shape
#     if bash "scripts/${script}"; then ...; return 0; fi
#     local rc=$?    # WRONG: this is the `if` statement's exit (always 0
#                    # when the if-condition was FALSE with no else)
# masked phase failures: a failed phase printed "[phase=failed] (exit 0)"
# AND the runner marched on to [phase=done] because `|| exit N` never fired.
# Now: capture rc on the same line as the command.

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
    echo "[phase=${tag}] start $(date -Iseconds)"
    uv run python "scripts/${script}"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[phase=${tag}] ok $(date -Iseconds)"
        return 0
    fi
    echo "[phase=failed] ${tag} (exit ${rc}) $(date -Iseconds)"
    return "$rc"
}

run_phase_py    preflight    i460_phase0_preflight.py    || exit 10
run_phase_py    rgen         i460_phase1_generate_R.py    || exit 11
run_phase_script train       i460_phase23_dispatch.sh     || exit 12
run_phase_script crosseval   i460_phase4_dispatch.sh      || exit 13

# Phase 5 runs on the VM after artifacts sync (it consumes JSON only).
# Pod-side runner stops at Phase 4 + emits the [phase=done] marker.
echo "[phase=done] phases 0..4 complete $(date -Iseconds)"
