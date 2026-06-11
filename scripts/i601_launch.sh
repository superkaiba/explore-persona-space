#!/usr/bin/env bash
# Task #601 — sequential gated pipeline driver (plan §10 launch contract).
#
# Runs, IN ORDER, aborting on any non-zero exit or failed gate:
#   1. Phase 0 zero-training reads  -> writes eval_results/issue_601/phase0/phase0_gate.json
#   2. gate: phase0_gate.json pass==true   (adapter-application HALT gate, plan §7 gate 2)
#   3. smoke: ONE FULL cell (ratio4to1_100p400n seed 42) + the §4 smoke asserts
#   4. gate: smoke sentinel records smoke_gate_pass==true
#   5. full sweep (all non-conditional cells x registered seeds)
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
uv run python - "$SLAB_ROOT/phase0/phase0_gate.json" <<'PY'
import json, sys
gate = json.loads(open(sys.argv[1]).read())
assert gate.get("pass") is True, f"phase0 gate FAILED: {gate}"
print(f"phase0 gate PASS (anchor_reuse_ok={gate.get('anchor_reuse_ok')}, "
      f"primary_space={gate.get('primary_space')})")
PY

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

echo "[phase=p5_sweep] $(date -u +%FT%TZ) launching full sweep (sub-log: $LOG_DIR/issue-601-sweep.log)"
# shellcheck disable=SC2086  # EXTRA_SWEEP_ARGS is a deliberate word-split passthrough
uv run python scripts/dispatch_neg_setpoint_601.py \
    --cells all --seeds 42,137 --n-gpus "$N_GPUS" --max-parallel "$N_GPUS" --resume \
    --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" $EXTRA_SWEEP_ARGS \
    > "$LOG_DIR/issue-601-sweep.log" 2>&1

echo "[phase=p6_final_sentinel] $(date -u +%FT%TZ) verifying final results sentinel"
test -f "$LOG_DIR/issue-601-results.json" || { echo "final sentinel missing"; exit 1; }

echo "[phase=done] $(date -u +%FT%TZ) issue-601 pipeline complete"
