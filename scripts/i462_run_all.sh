#!/usr/bin/env bash
# Issue #462 — pod-side end-to-end runner (Phases 0..4 on pod; Phase 5 on VM).
#
# Mirrors scripts/i460_run_all.sh: emits [phase=...] markers that
# poll_pipeline.py keys off, plus a 2-min heartbeat that keeps the main
# log mtime within poll's STALL_SEC threshold during long quiet phases.
#
# Phase tags use digit-free names (PHASE_RE = \[phase=([a-z_]+)).
#
# Pipeline phases:
#   preflight  — reuse i460 preflight (D_matrix, G_matrix, CONDITIONS, marker)
#   fetch_r    — download #460's frozen R from HF (regenerate fallback)
#   train      — 16 LoRAs × 4 epoch snapshots (epochs {1,2,3,5})
#   crosseval  — 4 levels × 16×16 cross-eval (per-level merger writes
#                G_logprob_matrix_ep{N}.json + cleans local adapter cache)
#   done
#
# rc-CAPTURE pattern (preserved from i460 round-2 fix): grab $? IMMEDIATELY
# after the command, NOT inside the `if !` branch. Previous shape masked
# phase failures and let the runner advance to [phase=done] with a failed
# phase.
#
# Launch:
#   nohup bash scripts/i462_run_all.sh > /workspace/logs/issue-462-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-462-run.pid
#
#   # Optional: skip the A1 smoke gate (we KNOW #460 implants at 5ep×300):
#   SKIP_SMOKE=1 nohup bash scripts/i462_run_all.sh > /workspace/logs/issue-462-run.log 2>&1 &

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

# Heartbeat: keep main-log mtime fresh during quiet single-condition trains.
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

# rc-CAPTURE pattern (round-2 fix #460):
#     if bash "scripts/${script}"; then ...; return 0; fi
#     local rc=$?    # WRONG: this is the `if` statement's exit (always 0
#                    # when if-condition was FALSE with no else)
# Now: capture rc on the same line as the command.

run_phase_script() {
    local tag="$1" script="$2" rc
    shift 2
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
    local tag="$1" script="$2" rc
    shift 2
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

# Phase 0: preflight (reuse #460's — same D_matrix, G_matrix, CONDITIONS,
# marker assert). #462 doesn't introduce new schema for any of these.
run_phase_py    preflight    i460_phase0_preflight.py    || exit 10

# Phase 1: fetch R (download #460's frozen R; fallback regenerate if HF
# download fails — see script header).
run_phase_py    fetch_r      i462_phase1_fetch_R.py      || exit 11

# Phase 2/3: train 16 LoRAs × 4 ckpts. Honors SKIP_SMOKE=1 to skip the A1
# smoke gate (#460 already validated 5ep×300 implants).
TRAIN_FLAGS=()
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
    TRAIN_FLAGS+=(--skip-smoke)
fi
run_phase_script train       i462_phase23_dispatch.sh    "${TRAIN_FLAGS[@]}" || exit 12

# Phase 4: crosseval per epoch level (1, 2, 3, 5). One dispatcher call
# per level; each call shards across GPUs 0+1 and runs the merger.
# Per-level adapter cache is wiped after each merger so disk stays bounded.
for EP in 1 2 3 5; do
    run_phase_script crosseval i462_phase4_dispatch.sh --adapter-epoch "$EP" || exit 13
done

# Phase 5 runs on the VM after artifact sync (consumes JSON only).
echo "[phase=done] phases 0..4 complete $(date -Iseconds)"
