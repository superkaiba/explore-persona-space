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
#
# Launch (RESUME — skip conds whose all 4 epoch adapters are already on HF):
#   nohup bash scripts/i474_run_all.sh --resume > /workspace/logs/issue-474-resume.log 2>&1 &
#   Use after a mid-sweep crash. The dispatcher checks each (arm, cond) via
#   scripts/i474_check_adapter_hf_presence.py and skips conds with all 4 of
#   {ep1, ep2, ep3, ep5} present; partial conds retrain fully (per-epoch
#   upload callback overwrites — no torn state).

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

SMOKE_MODE=0
RESUME_MODE=0
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE_MODE=1 ;;
        --resume) RESUME_MODE=1 ;;
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
# Phase 1 — DOWNLOAD #460's frozen R; NEVER regenerate / overwrite it.
# Round-3 fix: the v1/v2 run_all.sh called i460_phase1_generate_R.py which
# regenerates AND uploads to issue460_*/on_policy_R/, overwriting #460's
# archived artifact. i474_phase1_load_R.py downloads + verifies only.
run_phase_py    rload        i474_phase1_load_R.py       || exit 11

if [ "$SMOKE_MODE" -eq 1 ]; then
    # Unified-path smoke per plan v3 §4.10: A1 only x 1 ckpt (ep1).
    # Same dispatcher as the sweep, just --smoke-only on train and
    # --smoke --source-conds A1 on crosseval.
    #
    # Round-3 SMOKE-HARNESS fix: train_smoke trains ONLY A1 (both arms),
    # so crosseval_smoke MUST restrict the SOURCE loop to A1 — without
    # --source-conds A1 the eval would 404 on adapters/i474_pos_A5_ep1
    # (and every other untrained source). Targets (cid_j) still span
    # all 16 conditions on the trained A1 adapter -> 1 source x 16
    # targets x {pos,loc} at ep1 = 32 cells (exercises full KL top-K
    # eval + slot-read + tail-mass on real trained adapters).
    run_phase_script train_smoke i474_phase23_dispatch.sh --smoke-only || exit 12
    run_phase_script crosseval_smoke i474_phase4_dispatch.sh \
        --smoke --source-conds A1 --arms pos,loc --epochs 1 || exit 13
elif [ "$RESUME_MODE" -eq 1 ]; then
    # Round-5 FIX B — resume after mid-sweep crash. Train dispatcher
    # checks each (arm, cond) on HF for {ep1, ep2, ep3, ep5} and skips
    # conds with all 4 present. Partial conds retrain fully. Crosseval
    # runs full 16-source production matrix (resume on the eval side is
    # handled by phase4_eval's --resume which skips per-cell JSONs that
    # already landed atomically — pass it through).
    run_phase_script train_resume i474_phase23_dispatch.sh --resume || exit 12
    run_phase_script crosseval_resume i474_phase4_dispatch.sh --resume || exit 13
else
    run_phase_script train       i474_phase23_dispatch.sh     || exit 12
    run_phase_script crosseval   i474_phase4_dispatch.sh      || exit 13
fi

# Phase 5 runs on the VM after artifacts sync (it consumes JSON only).
echo "[phase=done] phases 0..4 complete $(date -Iseconds)"
