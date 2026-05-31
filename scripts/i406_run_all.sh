#!/usr/bin/env bash
# Issue #406 — full pod-side pipeline runner (Phases 1-3), chained with
# poll_pipeline.py-compatible [phase=...] markers + a heartbeat that keeps
# the log mtime fresh.
#
# Why this wrapper exists: the per-phase i406 dispatch scripts emit no
# [phase=...] lines and no done-sentinel, but scripts/poll_pipeline.py keys
# its done/stalled/dead decision off `[phase=done]` and a log-mtime stall
# threshold (STALL_SEC=900s). A single LoRA train runs ~20 min and writes
# its output to a per-condition log (not this main log), so without a
# heartbeat the poller would false-positive `stalled` and block a healthy
# run. This wrapper supplies both the phase markers and the heartbeat.
#
# Launched by the experimenter as:
#   nohup bash scripts/i406_run_all.sh > /workspace/logs/issue-406-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-406-run.pid
#
# Phases: 1 (divergence + cosine) -> 2 (16 LoRAs w/ A1 pilot gate;
#         C2 pilot + C2-C5 raw-format conds dropped 2026-05-31) ->
#         3 (cross-eval -> G matrix). Phase 4 (analysis + figures)
#         runs locally on the VM after this completes (it needs
#         pingouin and only consumes the JSON matrices, no GPU).
#
# Phase tags are digit-free on purpose: poll_pipeline.py's PHASE_RE is
# `\[phase=([a-z_]+)` and stops at digits.

export PATH="$HOME/.local/bin:$PATH"            # pod uv PATH gap (non-login shell)
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

# Heartbeat: keep main-log mtime < STALL_SEC (900s) during long quiet
# single-condition trains. [heartbeat] lines do not match PHASE_RE.
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

run_phase() {
    local tag="$1" script="$2"
    echo "[phase=${tag}] start $(date -Iseconds)"
    if bash "scripts/${script}"; then
        echo "[phase=${tag}] ok $(date -Iseconds)"
        return 0
    fi
    local rc=$?
    echo "[phase=failed] ${tag} (exit ${rc}) $(date -Iseconds)"
    return "${rc}"
}

run_phase divergence i406_phase1_dispatch.sh || exit 11
run_phase train      i406_phase2_dispatch.sh || exit 12
run_phase crosseval  i406_phase3_dispatch.sh || exit 13

echo "[phase=done] all phases complete $(date -Iseconds)"
