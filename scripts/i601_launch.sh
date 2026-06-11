#!/usr/bin/env bash
# Task #601 — sequential gated pipeline driver (plan §10 launch contract,
# hardened per plan v3 §D: self-daemonizing supervisor + heartbeat + cheap
# resume — launches 2 and 3 died at ssh-session teardown while their phase-0
# python child survived as an orphan).
#
# Runs, IN ORDER, aborting on any non-zero exit or failed gate:
#   1. Phase 0 zero-training reads  -> writes eval_results/issue_601/phase0/phase0_gate.json
#      (skip-cheap on re-run: parity-regime outputs are reused, the gate
#      recomputes from the existing JSONs — minutes, no GPU recompute)
#   2. gate: phase0_gate.json pass==true   (gate_schema 2: pass = Gate S
#      structural integrity ONLY — plan v3 §B; the HALT gate)
#      + capture anchor_reuse_ok (Gate A): false -> the sweep AUTOMATICALLY
#      gets --anchor-retrain-fallback (plan §4 Phase-0 item 3; concern
#      analyze-anchor-fallback-unwired — the fallback is enforced, not printed)
#   3. smoke: ONE FULL cell (ratio4to1_100p400n seed 42) + the §4 smoke asserts
#      (plan v3 §D item 4: SKIPPED when a prior sentinel already records
#      smoke_gate_pass==true — bare or .processed)
#   4. gate: smoke sentinel records smoke_gate_pass==true
#   5. full sweep (all non-conditional cells x registered seeds; round 4:
#      includes BOTH unconditional Phase-4 bridge cells — posonly_attn_lr5e6
#      AND posonly_alllinear_lr5e6, the true single-variable #471 lr-bridge)
#   6. Phase-4 bridge arrest classification over BOTH unconditional bridge
#      cells -> phase4/phase4a_verdict.json (scripts/i601_phase4_verdict.py;
#      plan §4 bands, seed-pooled per cell; routing call = any non-arrest)
#   7. conditional Phase-4b factor (posonly_attn_lr1e5 only): dispatched with
#      --cells phase4b ONLY on a bridge NON-ARREST verdict (the dispatcher
#      independently re-gates on the same sentinel); arrest/ambiguous -> 4b
#      skipped, reported open
#   8. final results sentinel check
#
# Launch (pod, repo at issue-601 HEAD, after the experimenter's preflight):
#   bash scripts/i601_launch.sh
# The script SELF-DAEMONIZES (plan v3 §D item 1): the launcher branch re-execs
# under `setsid --fork` (new session, no controlling TTY — ssh teardown/SIGHUP
# cannot reach the driver) and exits; the SUPERVISED process appends all
# output to $LOG_DIR/issue-601.log (the poller-pinned MAIN log) and writes its
# OWN pid ($$) to $LOG_DIR/issue-601.pid — never a wrapper's pid. A prefixed
# `nohup ... &` still works and is now redundant. Relaunch while the pid-file
# pid is alive refuses with exit 3. Any driver death is recovered by
# re-running the same launch line at zero GPU recompute cost (item 4).
#
# Logging contract (poll_pipeline.py): each sub-step's verbose output goes to
# its OWN sub-log; the SUPERVISED process's stdout is the main pipeline log
# ($LOG_DIR/issue-601.log) and carries one [phase=...] line per step, a
# 120-second `[hb] <utc> pid=<driver> phase=<step>` heartbeat (plan v3 §D
# item 3; driver dead <=> no [hb] for >5 min AND the pid-file pid not alive),
# `[phase=abort] rc=<rc>` on any non-zero exit (single combined EXIT trap),
# plus the SINGLE terminal [phase=done] line — sub-steps must never leak
# their own "[phase=done]" into the main log (incident #545: a mid-run done
# token produced a false status=done).
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

MAIN_LOG="$LOG_DIR/issue-601.log"     # poller + handle sidecar are pinned here
PID_FILE="$LOG_DIR/issue-601.pid"
PHASE_FILE="$LOG_DIR/issue-601.phase"

# ── Plan v3 §D items 1+2: self-daemonization + relaunch guard ────────────────
if [ -z "${I601_SUPERVISED:-}" ]; then
    # §F assumption 20: util-linux setsid with --fork must exist on the image.
    if ! setsid --version >/dev/null 2>&1; then
        echo "[launcher] FATAL: 'setsid --version' failed — util-linux setsid (with --fork) required (plan v3 §F assumption 20)"
        exit 4
    fi
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
        echo "[launcher] already running pid=$(cat "$PID_FILE") — refusing relaunch (exit 3)"
        exit 3
    fi
    export I601_SUPERVISED=1
    # New session + new process group + no controlling TTY: ssh teardown /
    # SIGHUP cannot reach the supervised driver regardless of invocation.
    setsid --fork bash "$0" "$@" >> "$MAIN_LOG" 2>&1 < /dev/null
    echo "[launcher] detached supervised driver; main log: $MAIN_LOG; pid file: $PID_FILE"
    exit 0
fi

# ── SUPERVISED branch: own-pid file, heartbeat, combined EXIT trap ───────────
# Supervised-side relaunch guard (closes the launcher-check -> pid-write race
# on a rapid double dispatch): refuse if a DIFFERENT live pid holds the file.
if [ -f "$PID_FILE" ]; then
    _old_pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$_old_pid" ] && [ "$_old_pid" != "$$" ] && kill -0 "$_old_pid" 2>/dev/null; then
        echo "[launcher] already running pid=$_old_pid — refusing relaunch (exit 3, supervised guard)"
        exit 3
    fi
fi
echo $$ > "$PID_FILE"   # the SUPERVISED process's OWN pid — never a wrapper's

# CURRENT_PHASE is exported before each step (plan v3 §D item 3). The
# heartbeat subshell forked below CANNOT see later exports (env is copied at
# fork), so set_phase ALSO mirrors the phase to $PHASE_FILE, which the
# heartbeat re-reads each tick — the [hb] line always carries the live phase.
set_phase() {
    CURRENT_PHASE="$1"
    export CURRENT_PHASE
    printf '%s' "$1" > "$PHASE_FILE"
}
set_phase init

HB_INTERVAL="${I601_HB_INTERVAL:-120}"
(
    # Inside a bash subshell $$ is the PARENT's pid — i.e. the supervised
    # driver pid, matching $PID_FILE (use $BASHPID for the subshell itself).
    while true; do
        echo "[hb] $(date -u +%FT%TZ) pid=$$ phase=$(cat "$PHASE_FILE" 2>/dev/null || echo unset)"
        sleep "$HB_INTERVAL"
    done
) &
HB_PID=$!

on_exit() {
    rc=$?
    # Deterministic heartbeat teardown: freeze the subshell so it cannot fork
    # a fresh sleep, reap the IN-FLIGHT sleep child (it survives a bare kill
    # of the subshell — reparented to init it keeps the stdout fd open), then
    # resume + terminate the subshell itself.
    kill -STOP "$HB_PID" 2>/dev/null || true
    pkill -P "$HB_PID" 2>/dev/null || true
    kill -CONT "$HB_PID" 2>/dev/null || true
    kill "$HB_PID" 2>/dev/null || true
    if [ "$rc" -ne 0 ]; then
        echo "[phase=abort] rc=$rc"
    fi
}
trap on_exit EXIT

# Preamble selftest hook (fixture-level unit exercise of the detachment /
# guard / heartbeat / trap contract — tests/test_i601_round7_amendment.py).
# NEVER set in production: short-circuits before any pipeline step.
if [ -n "${I601_LAUNCH_SELFTEST:-}" ]; then
    set_phase p_selftest
    echo "[phase=p_selftest] supervised preamble selftest (pid=$$)"
    sleep "${I601_SELFTEST_SLEEP:-3}"
    rc="${I601_SELFTEST_RC:-0}"
    if [ "$rc" = "0" ]; then
        echo "[phase=done] $(date -u +%FT%TZ) selftest complete"
    fi
    exit "$rc"
fi

set_phase p1_phase0
echo "[phase=p1_phase0] $(date -u +%FT%TZ) launching Phase 0 reads (sub-log: $LOG_DIR/issue-601-phase0.log; skip-cheap on re-run)"
uv run python scripts/i601_phase0_reads.py --n-gpus "$N_GPUS" \
    --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" \
    > "$LOG_DIR/issue-601-phase0.log" 2>&1

set_phase p2_phase0_gate
echo "[phase=p2_phase0_gate] $(date -u +%FT%TZ) checking phase0_gate.json (gate_schema 2: pass = Gate S only)"
# Diagnostics go to stderr (the main log); ONLY the anchor_reuse_ok value is
# captured on stdout so the fallback routing below is code, not prose.
ANCHOR_OK=$(uv run python - "$SLAB_ROOT/phase0/phase0_gate.json" <<'PY'
import json, sys
gate = json.loads(open(sys.argv[1]).read())
assert gate.get("pass") is True, f"phase0 gate FAILED (Gate S): {gate}"
print(
    f"phase0 Gate S PASS (gate_schema={gate.get('gate_schema')}, "
    f"anchor_reuse_ok={gate.get('anchor_reuse_ok')}, "
    f"primary_space={gate.get('primary_space')})",
    file=sys.stderr,
)
print("true" if gate.get("anchor_reuse_ok") else "false")
PY
)
ANCHOR_FALLBACK_ARGS=""
if [ "$ANCHOR_OK" != "true" ]; then
    ANCHOR_FALLBACK_ARGS="--anchor-retrain-fallback"
    echo "[phase=p2_phase0_gate] anchor_reuse_ok=false (Gate A) -> sweep gets $ANCHOR_FALLBACK_ARGS (plan §4 Phase-0 item 3)"
else
    echo "[phase=p2_phase0_gate] anchor_reuse_ok=true -> parent anchor reused"
fi

set_phase p3_smoke
# Plan v3 §D item 4: a prior PASSing smoke sentinel (bare or .processed —
# same accept-rename logic as p4) makes the smoke a skip on relaunch.
SMOKE_SKIP=$(uv run python - "$LOG_DIR/issue-601-smoke-results.json" <<'PY'
import json, pathlib, sys
bare = pathlib.Path(sys.argv[1])
candidate = bare if bare.exists() else bare.with_suffix(".json.processed")
ok = False
if candidate.exists():
    try:
        payload = json.loads(candidate.read_text())
        note = json.loads(payload.get("note") or payload.get("payload") or "{}")
        ok = note.get("smoke_gate_pass") is True
    except (OSError, json.JSONDecodeError) as exc:
        print(f"smoke sentinel unreadable ({exc}); re-running the smoke", file=sys.stderr)
print("skip" if ok else "run")
PY
)
if [ "$SMOKE_SKIP" = "skip" ]; then
    echo "[phase=p3_smoke] sentinel valid; skip"
else
    echo "[phase=p3_smoke] $(date -u +%FT%TZ) launching smoke (ONE FULL cell; sub-log: $LOG_DIR/issue-601-smoke.log)"
    uv run python scripts/dispatch_neg_setpoint_601.py \
        --cells ratio4to1_100p400n --seeds 42 --smoke --n-gpus "$N_GPUS" \
        --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" \
        > "$LOG_DIR/issue-601-smoke.log" 2>&1
fi

set_phase p4_smoke_gate
echo "[phase=p4_smoke_gate] $(date -u +%FT%TZ) checking smoke sentinel"
# Accept the poller's .processed rename: the dispatcher writes the smoke
# sentinel BEFORE its post-sentinel HF raw-completions upload, so
# poll_pipeline.py can legitimately drain + rename it in that window.
# Mirrors _check_gates (dispatch_neg_setpoint_601.py) and p7/p8 below —
# under `set -euo pipefail` a bare-name-only read here would abort the
# pipeline AFTER a successful smoke (round-3 blocker
# smoke-sentinel-processed-race).
uv run python - "$LOG_DIR/issue-601-smoke-results.json" <<'PY'
import json, pathlib, sys
bare = pathlib.Path(sys.argv[1])
candidate = bare if bare.exists() else bare.with_suffix(".json.processed")
if not candidate.exists():
    raise SystemExit(f"smoke gate FAILED: sentinel missing at {bare} (also checked {candidate})")
payload = json.loads(candidate.read_text())
note = json.loads(payload.get("note") or payload.get("payload") or "{}")
assert note.get("smoke_gate_pass") is True, f"smoke gate FAILED ({candidate}): {note}"
print(f"smoke gate PASS (sentinel: {candidate})")
PY

set_phase p5_sweep
echo "[phase=p5_sweep] $(date -u +%FT%TZ) launching full sweep (sub-log: $LOG_DIR/issue-601-sweep.log; fallback args: '${ANCHOR_FALLBACK_ARGS}')"
# shellcheck disable=SC2086  # EXTRA_SWEEP_ARGS/ANCHOR_FALLBACK_ARGS are deliberate word-split passthroughs
uv run python scripts/dispatch_neg_setpoint_601.py \
    --cells all --seeds 42,137 --n-gpus "$N_GPUS" --max-parallel "$N_GPUS" --resume \
    --slab-root "$SLAB_ROOT" --log-dir "$LOG_DIR" $ANCHOR_FALLBACK_ARGS $EXTRA_SWEEP_ARGS \
    > "$LOG_DIR/issue-601-sweep.log" 2>&1

set_phase p6_phase4a_verdict
echo "[phase=p6_phase4a_verdict] $(date -u +%FT%TZ) classifying Phase-4 bridge arrest (attn@5e6 + alllinear@5e6; sub-log: $LOG_DIR/issue-601-phase4a-verdict.log)"
uv run python scripts/i601_phase4_verdict.py --slab-root "$SLAB_ROOT" \
    > "$LOG_DIR/issue-601-phase4a-verdict.log" 2>&1
PHASE4A_CALL=$(uv run python - "$SLAB_ROOT/phase4/phase4a_verdict.json" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1]).read()).get("call"))
PY
)
echo "[phase=p6_phase4a_verdict] bridge call=$PHASE4A_CALL"

set_phase p7_phase4b
if [ "$PHASE4A_CALL" = "non-arrest" ]; then
    echo "[phase=p7_phase4b] $(date -u +%FT%TZ) bridge NON-ARREST -> dispatching conditional 4b factor posonly_attn_lr1e5 (sub-log: $LOG_DIR/issue-601-phase4b.log)"
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
    echo "[phase=p7_phase4b] bridge call=$PHASE4A_CALL -> 4b uninformative, SKIPPED (recorded in phase4a_verdict.json; reported open per plan §4/§7)"
fi

set_phase p8_final_sentinel
echo "[phase=p8_final_sentinel] $(date -u +%FT%TZ) verifying final results sentinel"
# Accept the poller's .processed rename: the main sweep's sentinel lands at
# p5-end and poll_pipeline.py can legitimately process it while p6/p7 run.
test -f "$LOG_DIR/issue-601-results.json" \
    || test -f "$LOG_DIR/issue-601-results.json.processed" \
    || { echo "final sentinel missing"; exit 1; }

echo "[phase=done] $(date -u +%FT%TZ) issue-601 pipeline complete"
