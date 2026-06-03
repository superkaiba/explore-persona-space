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

# Phase B (conditional).
PHASE_B_FIRED=0
if [ "$ANCHOR_STEP" = "LOCKSTEP" ]; then
    echo "=== [phase=phaseB_SKIPPED] LOCKSTEP -- Phase B does NOT fire ==="
else
    echo "=== [phase=phaseB] anchor_step=$ANCHOR_STEP $(date -Iseconds) ==="
    bash scripts/i471_phaseB_sweep.sh --anchor-step "$ANCHOR_STEP"
    PHASE_B_FIRED=1
fi

# Phase 4 eval set: always include cond1_withneg + cond1_posonly + 4 #465
# baselines; add 3 cond2_* arms when Phase B fired. Note the cond1_withneg
# adapter name uses --anchor-step from Phase A so the eval picks up the
# chosen checkpoint, not Phase A's intermediate step dirs.
WITHNEG_RUN=i471_route_a_cond1_withneg
POSONLY_RUN=i471_route_a_cond1_posonly
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

# Phase 5 analyzer + figures.
echo "=== [phase=phase5] $(date -Iseconds) ==="
uv run python scripts/i471_phase5_analyze.py \
    > "$LOG_DIR/phase5_route_a.log" 2>&1 || {
    echo "WARN: phase5 exited non-zero (figures may be partial); continuing." >&2
}
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
