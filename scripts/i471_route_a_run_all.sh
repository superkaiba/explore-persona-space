#!/usr/bin/env bash
# Plan v3 §4.1 -- top-level route-(a) orchestrator (pod-side launcher).
#
# Sequencing:
#   Phase 0: preflight + R generations (skip-r-gen if Phase 0 already ran).
#   Phase A: i471_phaseA_anchor.sh runs cond1_withneg then cond1_posonly.
#   phaseA analyze: pick anchor step (or report lockstep).
#   Phase B (CONDITIONAL on anchor): cond2_k0 / cond2_k1 / cond2_k3 at
#     max_steps=s* (with-negatives recipe).
#   Phase 4: i471_phase4_eval.py --free-gen-emission on the eval adapter
#     set (always cond1_withneg + cond1_posonly + 4 #465 baselines; +3
#     Phase B arms when fired).
#   Phase 5: analyzer + figures.
#
# Failure semantics: any phase failure writes a sentinel for
# poll_pipeline.py and exits non-zero. Lockstep is NOT a failure -- Phase
# B is skipped, Phase 4 still runs on the two cond1 anchor checkpoints
# (so the disentanglement read survives under the lockstep finding).

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_471
mkdir -p "$LOG_DIR"

# Round-3 code-review BLOCKER fix: clear ANY stale issue-471 success/failure
# sentinels at launcher start, BEFORE Phase 0. `poll_pipeline.py` drains every
# matching `/workspace/logs/issue-471-epm_results-*.json` it sees; a failed
# rerun that leaves an OLD success sentinel from a prior attempt makes the
# orchestrator advance as if THIS run completed. Only a sentinel written by
# THIS invocation may be observable.
mkdir -p /workspace/logs
for _stale in /workspace/logs/issue-471-epm_results-*.json \
              /workspace/logs/issue-471-epm_results_lockstep-*.json \
              /workspace/logs/issue-471-epm_results_smoke_failed-*.json; do
    if [ -e "$_stale" ]; then
        echo "clearing stale sentinel: $_stale"
        rm -f "$_stale"
    fi
done

SKIP_PHASE0=0
SKIP_R_GEN=0
for arg in "$@"; do
    case "$arg" in
        --skip-phase0) SKIP_PHASE0=1 ;;
        --skip-r-gen) SKIP_R_GEN=1 ;;
    esac
done

# Phase 0 (idempotent if --skip-r-gen).
if [ "$SKIP_PHASE0" -eq 0 ]; then
    echo "=== [phase=phase0_route_a] $(date -Iseconds) ==="
    if [ "$SKIP_R_GEN" -eq 1 ]; then
        uv run python scripts/i471_phase0_preflight.py --skip-r-gen \
            > "$LOG_DIR/phase0_route_a.log" 2>&1
    else
        uv run python scripts/i471_phase0_preflight.py \
            > "$LOG_DIR/phase0_route_a.log" 2>&1
    fi
    echo "phase0 OK (see $LOG_DIR/phase0_route_a.log)"
else
    echo "=== [phase=phase0_route_a SKIPPED] ==="
fi

# Phase A (BOTH cond1 variants).
echo "=== [phase=phaseA] $(date -Iseconds) ==="
bash scripts/i471_phaseA_anchor.sh

# phaseA analyzer (deterministic; reads training logs).
echo "=== [phase=phaseA_analyze] $(date -Iseconds) ==="
uv run python scripts/i471_phaseA_analyze.py \
    > "$LOG_DIR/phaseA_analyze.log" 2>&1
echo "phaseA_analyze OK (see $LOG_DIR/phaseA_analyze.log)"

# Read the anchor step (or lockstep flag) out of the analyzer's JSON.
ANCHOR_JSON=eval_results/issue_471/route_a/phaseA_anchor.json
if [ ! -f "$ANCHOR_JSON" ]; then
    echo "FATAL: $ANCHOR_JSON missing after phaseA_analyze." >&2
    exit 2
fi
ANCHOR_STEP=$(uv run python -c "
import json
d = json.load(open('$ANCHOR_JSON'))
print(d.get('anchor_step') if d.get('anchor_step') is not None else 'LOCKSTEP')
")
MATCHED_POSONLY_STEP=$(uv run python -c "
import json
d = json.load(open('$ANCHOR_JSON'))
print(d.get('matched_posonly_step'))
")
echo "phaseA result: anchor_step=$ANCHOR_STEP matched_posonly_step=$MATCHED_POSONLY_STEP"

# Phase B (conditional). Phase B's i471_phaseB_sweep.sh ALREADY trains each
# cond2_* arm under run_name=i471_route_a_<cond>_step${ANCHOR_STEP}, so the
# Phase B output directories already carry the _step suffix natively. Only
# the cond1 variants need the post-train mirror (their training run_name is
# the bare slug; the chosen step is a sub-directory under adapters/).
PHASE_B_FIRED=0
if [ "$ANCHOR_STEP" = "LOCKSTEP" ]; then
    echo "=== [phase=phaseB_SKIPPED] LOCKSTEP -- Phase B does NOT fire ==="
else
    echo "=== [phase=phaseB] anchor_step=$ANCHOR_STEP $(date -Iseconds) ==="
    bash scripts/i471_phaseB_sweep.sh --anchor-step "$ANCHOR_STEP"
    PHASE_B_FIRED=1
fi

# Plan v3 §4.5 / round-2 code-review BLOCKER fix.
# Mirror the chosen anchor checkpoint dirs of BOTH cond1 variants into
# stepped adapter_id paths (adapters/<run_name>_step<step>/) AND upload them
# to HF for durability + reproducibility. Phase 4's _download_adapters
# prefers the local mirror; the HF upload is the safety net for re-runs
# from a fresh pod. Stepped naming is consistent with how phaseB_sweep.sh
# names its outputs and how phase5_analyze.py reads them back.
if [ "$ANCHOR_STEP" = "LOCKSTEP" ]; then
    # Lockstep regime: there is no s* satisfying the anchor rule. Plan v3
    # §4.3 says still run Phase 4 on both cond1 variants at their FINAL
    # SAVED step (NOT their final trajectory step) so the disentanglement
    # read survives. Trajectory rows fire at log_every=5 but checkpoints
    # save at save_steps=10, so we MUST read the analyzer's
    # `withneg_final_saved_step` (floored to a checkpoint multiple) — the
    # max trajectory step itself can be 75 while the last saved checkpoint
    # is 70, in which case `i471_upload_anchor_adapter.py` would fail on
    # the missing checkpoint dir before Phase 4 ever runs (round-3
    # code-review BLOCKER fix).
    WITHNEG_STEP=$(uv run python -c "
import json
d = json.load(open('$ANCHOR_JSON'))
print(d.get('withneg_final_saved_step', 0))
")
    if [ -z "$WITHNEG_STEP" ] || [ "$WITHNEG_STEP" = "0" ]; then
        echo "FATAL: lockstep but no cond1_withneg SAVED checkpoint to mirror." >&2
        exit 3
    fi
else
    WITHNEG_STEP="$ANCHOR_STEP"
fi
POSONLY_STEP="$MATCHED_POSONLY_STEP"
# Fail-loud pre-mirror check: both checkpoint dirs MUST exist on disk before
# the mirror runs. If they don't, the analyzer chose a step at a non-saved
# multiple — surface that immediately, don't let `i471_upload_anchor_adapter.py`
# die mid-stream with a less obvious traceback (round-3 code-review BLOCKER fix).
for _pair in "i471_route_a_cond1_withneg:${WITHNEG_STEP}" "i471_route_a_cond1_posonly:${POSONLY_STEP}"; do
    _run="${_pair%%:*}"
    _step="${_pair##*:}"
    if [ ! -d "adapters/${_run}/checkpoint-${_step}" ]; then
        echo "FATAL: missing checkpoint dir adapters/${_run}/checkpoint-${_step}." >&2
        echo "       Analyzer picked a step that does NOT match save_steps." >&2
        ls -d "adapters/${_run}/checkpoint-"* 2>/dev/null | sed 's/^/  candidate: /' >&2 || true
        exit 4
    fi
done
echo "=== [phase=mirror_anchor] withneg=step${WITHNEG_STEP} posonly=step${POSONLY_STEP} $(date -Iseconds) ==="
uv run python scripts/i471_upload_anchor_adapter.py \
    --run-name i471_route_a_cond1_withneg \
    --step "$WITHNEG_STEP" \
    --extra "i471_route_a_cond1_posonly:${POSONLY_STEP}" \
    > "$LOG_DIR/mirror_anchor.log" 2>&1
echo "mirror_anchor OK (see $LOG_DIR/mirror_anchor.log)"

# Phase 4 eval set: cond1_withneg_step<W> + cond1_posonly_step<P> + 4 #465
# baselines; add 3 cond2_*_step<A> arms when Phase B fired. Every adapter id
# below is a directory that exists at adapters/<id>/ on the pod (Phase A/B
# wrote it natively or i471_upload_anchor_adapter.py mirrored it).
WITHNEG_RUN="i471_route_a_cond1_withneg_step${WITHNEG_STEP}"
POSONLY_RUN="i471_route_a_cond1_posonly_step${POSONLY_STEP}"
ADAPTERS="$WITHNEG_RUN $POSONLY_RUN i465_cond1 i465_cond2_k0 i465_cond2_k1 i465_cond2_k3"
if [ "$PHASE_B_FIRED" -eq 1 ]; then
    ADAPTERS="$ADAPTERS i471_route_a_cond2_k0_step${ANCHOR_STEP}"
    ADAPTERS="$ADAPTERS i471_route_a_cond2_k1_step${ANCHOR_STEP}"
    ADAPTERS="$ADAPTERS i471_route_a_cond2_k3_step${ANCHOR_STEP}"
fi
echo "=== [phase=phase4] eval adapters: $ADAPTERS  $(date -Iseconds) ==="
# shellcheck disable=SC2086
uv run python scripts/i471_phase4_eval.py \
    --adapters $ADAPTERS \
    --free-gen-emission \
    > "$LOG_DIR/phase4_route_a.log" 2>&1
echo "phase4 OK (see $LOG_DIR/phase4_route_a.log)"

# Phase 5 analyzer + figures. A Phase 5 failure must FAIL the run -- it is
# the last gate before the success sentinel and silently writing "complete"
# after partial figures would mask the failure from poll_pipeline.py.
echo "=== [phase=phase5] $(date -Iseconds) ==="
uv run python scripts/i471_phase5_analyze.py \
    --withneg-adapter "$WITHNEG_RUN" \
    --posonly-adapter "$POSONLY_RUN" \
    > "$LOG_DIR/phase5_route_a.log" 2>&1
echo "phase5 OK (see $LOG_DIR/phase5_route_a.log)"

echo "=== [phase=done] $(date -Iseconds) ==="

# End-of-run sentinel for poll_pipeline.py.
SENTINEL="/workspace/logs/issue-471-epm_results-$(date +%s).json"
mkdir -p "$(dirname "$SENTINEL")"
uv run python - <<EOF
import json, datetime
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 471,
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": (
        "Route-(a) Phase 0 -> A -> (B?) -> 4 -> 5 complete. "
        "phaseA anchor_step=$ANCHOR_STEP, phase_B_fired=$PHASE_B_FIRED. "
        "See eval_results/issue_471/route_a/ + figures/issue_471/."
    ),
    "phaseA_anchor_step": "$ANCHOR_STEP",
    "phaseB_fired": $PHASE_B_FIRED,
}
with open("$SENTINEL", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote results sentinel: $SENTINEL")
EOF
