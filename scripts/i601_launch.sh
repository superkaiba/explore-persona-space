#!/usr/bin/env bash
# Task #601 — sequential gated pipeline driver (plan §10 launch contract).
#
# Runs, IN ORDER, aborting on any non-zero exit or failed gate:
#   1. Phase 0 zero-training reads  -> writes eval_results/issue_601/phase0/phase0_gate.json
#   2. gate: phase0_gate.json pass==true   (adapter-application HALT gate, plan §7 gate 2)
#      + capture anchor_reuse_ok: false -> the sweep AUTOMATICALLY gets
#      --anchor-retrain-fallback (plan §4 Phase-0 item 3; concern
#      analyze-anchor-fallback-unwired — the fallback is enforced, not printed)
#   3. smoke: ONE FULL cell (ratio4to1_100p400n seed 42) + the §4 smoke asserts
#   4. gate: smoke sentinel records smoke_gate_pass==true
#   5. full sweep (all non-conditional cells x registered seeds)
#   6. Phase-4a arrest classification -> phase4/phase4a_verdict.json
#      (scripts/i601_phase4_verdict.py; plan §4 bands, seed-pooled)
#   7. conditional Phase-4b factorization: dispatched with --cells phase4b
#      ONLY on a 4a NON-ARREST verdict (the dispatcher independently re-gates
#      on the same sentinel); arrest/ambiguous -> 4b skipped, reported open
#   8. final results sentinel check
#
# Launch (pod, repo at issue-601 HEAD, after the experimenter's preflight):
#   nohup bash scripts/i601_launch.sh > /workspace/logs/issue-601-pipeline.log 2>&1 < /dev/null &
#
# Logging contract (poll_pipeline.py): each sub-step's verbose output goes to
# its OWN sub-log; THIS script's stdout is the main pipeline log and carries
# one [phase=...] line per step plus the SINGLE terminal [phase=done] line —
# sub-steps must never leak their own "[phase=done]" into the main log
# (incident #545: a mid-run done token produced a false status=done).
#
# NOTE on preflight: this driver does NOT re-run orchestrate.preflight — the
# experimenter runs it pre-launch. If you add it here, parse `preflight
# --json` and tolerate ONLY the documented feature-branch false positive
# ("Local is N commit(s) behind origin/main"); a bare `preflight || exit`
# under set -e kills every issue-branch launch (incident #552).
set -euo pipefail

LOG_DIR="${LOG_DIR:-/workspace/logs}"
SLAB_ROOT="${SLAB_ROOT:-eval_results/issue_601}"
N_GPUS="${N_GPUS:-4}"
EXTRA_SWEEP_ARGS="${EXTRA_SWEEP_ARGS:-}"
mkdir -p "$LOG_DIR"

echo "[phase=p1_phase0] $(date -u +%FT%TZ) launching Phase 0 reads (sub-log: $LOG_DIR/issue-601-phase0.log)"
uv run python scripts/i601_phase0_reads.py --n-gpus "$N_GPUS" \
    --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" \
    > "$LOG_DIR/issue-601-phase0.log" 2>&1

echo "[phase=p2_phase0_gate] $(date -u +%FT%TZ) checking phase0_gate.json"
# Diagnostics go to stderr (the main log); ONLY the anchor_reuse_ok value is
# captured on stdout so the fallback routing below is code, not prose.
ANCHOR_OK=$(uv run python - "$SLAB_ROOT/phase0/phase0_gate.json" <<'PY'
import json, sys
gate = json.loads(open(sys.argv[1]).read())
assert gate.get("pass") is True, f"phase0 gate FAILED: {gate}"
print(
    f"phase0 gate PASS (anchor_reuse_ok={gate.get('anchor_reuse_ok')}, "
    f"primary_space={gate.get('primary_space')})",
    file=sys.stderr,
)
print("true" if gate.get("anchor_reuse_ok") else "false")
PY
)
ANCHOR_FALLBACK_ARGS=""
if [ "$ANCHOR_OK" != "true" ]; then
    ANCHOR_FALLBACK_ARGS="--anchor-retrain-fallback"
    echo "[phase=p2_phase0_gate] anchor_reuse_ok=false -> sweep gets $ANCHOR_FALLBACK_ARGS (plan §4 Phase-0 item 3)"
else
    echo "[phase=p2_phase0_gate] anchor_reuse_ok=true -> parent anchor reused"
fi

echo "[phase=p3_smoke] $(date -u +%FT%TZ) launching smoke (ONE FULL cell; sub-log: $LOG_DIR/issue-601-smoke.log)"
uv run python scripts/dispatch_neg_setpoint_601.py \
    --cells ratio4to1_100p400n --seeds 42 --smoke --n-gpus "$N_GPUS" \
    --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" \
    > "$LOG_DIR/issue-601-smoke.log" 2>&1

echo "[phase=p4_smoke_gate] $(date -u +%FT%TZ) checking smoke sentinel"
uv run python - "$LOG_DIR/issue-601-smoke-results.json" <<'PY'
import json, sys
payload = json.loads(open(sys.argv[1]).read())
note = json.loads(payload.get("note") or "{}")
assert note.get("smoke_gate_pass") is True, f"smoke gate FAILED: {note}"
print("smoke gate PASS")
PY

echo "[phase=p5_sweep] $(date -u +%FT%TZ) launching full sweep (sub-log: $LOG_DIR/issue-601-sweep.log; fallback args: '${ANCHOR_FALLBACK_ARGS}')"
# shellcheck disable=SC2086  # EXTRA_SWEEP_ARGS/ANCHOR_FALLBACK_ARGS are deliberate word-split passthroughs
uv run python scripts/dispatch_neg_setpoint_601.py \
    --cells all --seeds 42,137 --n-gpus "$N_GPUS" --max-parallel "$N_GPUS" --resume \
    --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" $ANCHOR_FALLBACK_ARGS $EXTRA_SWEEP_ARGS \
    > "$LOG_DIR/issue-601-sweep.log" 2>&1

echo "[phase=p6_phase4a_verdict] $(date -u +%FT%TZ) classifying Phase-4a arrest (sub-log: $LOG_DIR/issue-601-phase4a-verdict.log)"
uv run python scripts/i601_phase4_verdict.py --slab-root "$SLAB_ROOT" \
    > "$LOG_DIR/issue-601-phase4a-verdict.log" 2>&1
PHASE4A_CALL=$(uv run python - "$SLAB_ROOT/phase4/phase4a_verdict.json" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1]).read()).get("call"))
PY
)
echo "[phase=p6_phase4a_verdict] 4a call=$PHASE4A_CALL"

if [ "$PHASE4A_CALL" = "non-arrest" ]; then
    echo "[phase=p7_phase4b] $(date -u +%FT%TZ) 4a NON-ARREST -> dispatching conditional 4b cells (sub-log: $LOG_DIR/issue-601-phase4b.log)"
    # shellcheck disable=SC2086
    uv run python scripts/dispatch_neg_setpoint_601.py \
        --cells phase4b --seeds 42,137 --n-gpus "$N_GPUS" --max-parallel "$N_GPUS" --resume \
        --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" \
        --sentinel-name issue-601-phase4b-results.json $EXTRA_SWEEP_ARGS \
        > "$LOG_DIR/issue-601-phase4b.log" 2>&1
    # Accept the poller's .processed rename (poll_pipeline.py may process a
    # sentinel between the dispatch exit and this check).
    test -f "$LOG_DIR/issue-601-phase4b-results.json" \
        || test -f "$LOG_DIR/issue-601-phase4b-results.json.processed" \
        || { echo "phase4b sentinel missing"; exit 1; }
else
    echo "[phase=p7_phase4b] 4a call=$PHASE4A_CALL -> 4b uninformative, SKIPPED (recorded in phase4a_verdict.json; reported open per plan §4/§7)"
fi

echo "[phase=p8_final_sentinel] $(date -u +%FT%TZ) verifying final results sentinel"
# Accept the poller's .processed rename: the main sweep's sentinel lands at
# p5-end and poll_pipeline.py can legitimately process it while p6/p7 run.
test -f "$LOG_DIR/issue-601-results.json" \
    || test -f "$LOG_DIR/issue-601-results.json.processed" \
    || { echo "final sentinel missing"; exit 1; }

echo "[phase=done] $(date -u +%FT%TZ) issue-601 pipeline complete"
