#!/usr/bin/env bash
# Issue #2162 — pod dispatcher for scripts/issue2162_run.py.
#
# Forked from scripts/issue2094_dispatch.sh. Phases:
#   import-check    deferred-import resolution (CPU, no GPU)
#   bank            P1: v_ce/v_pe bank + degeneracy guard + injection gate (GPU 0)
#   anchors         P2: unpatched anchors, SHARDED across every visible GPU
#   pilot           P3-entry: ONE production-shape timed block (GPU 0; rc=22 refusal)
#   grid            P3: 234-block claim-file queue, one worker per visible GPU
#   margin          pools-dependent margin TF legs, one worker per visible GPU
#   stage2          P4: layer x dose survivor grid (issue2162_stage2.py), fanned out
#   upload          P5: bulk HF uploads + pod sentinel (CPU)
#   all             import-check -> bank -> anchors -> pilot -> gate3 -> grid
#                   -> margin (OPPORTUNISTIC) -> upload
#
# Margin semantics (r2 MAJOR 1, superseding the r1 C4 literal wiring):
# the pools file is judge-built from the ~28k-call Batch-API behavior wave
# (2-24h calendar SLA), so the `all` chain must NEVER park the 8-GPU pod
# behind it (#664 idle-burn). Pools staged in time -> margin rides the wide
# pod here; absent -> the margin leg is DEFERRED LOUDLY: upload + sentinel +
# teardown proceed, the sentinel carries `margin_deferred: true` plus the
# deferred-leg recipe (issue2162_run._margin_state), and the deferred leg
# runs later on a fresh 1x H100 once pools land:
#   uv run python scripts/issue2162_judge.py --phase pools   (VM-side)
#   scp pools.json to the pod, then: dispatch.sh margin && dispatch.sh upload
# The STANDALONE `margin` phase keeps the rc=24 HARD HALT — an explicitly
# requested margin with no pools is an error, never a silent skip.
#
# Worker count is DERIVED from the realized GPU count (`nvidia-smi -L`) at
# launch — never hardcoded — so a 4-GPU fallback pod re-shards with no code
# change (plan §4.6 mechanical gate 2). Each worker gets CUDA_VISIBLE_DEVICES
# pinned in ITS OWN launcher env (never `+gpu_id=N`; the in-process clobber is
# defeated by import-time cuInit — gotchas.md).
#
# Grid/margin workers pull blocks from the SHARED claim-file queue inside the
# driver; the dispatcher only fans out N identical workers.
#
# The single terminal `[phase=done]` line is emitted ONLY at the very end of a
# successful dispatch (the poller keys on it).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

# GCE lane exports tokens via startup script and has NO .env — source conditionally.
if [ -f ./.env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

PHASE="${1:-all}"
shift || true

DRIVER="scripts/issue2162_run.py"
STAGE2_DRIVER="scripts/issue2162_stage2.py"
OUT_ROOT="${EPM_2162_OUT_ROOT:-/workspace/issue2162_out}"
LOG_DIR="${EPM_2162_LOG_DIR:-/workspace/logs}"
POOLS_PATH="${EPM_2162_POOLS:-$OUT_ROOT/pools.json}"
BEST_CELLS_PATH="${EPM_2162_BEST_CELLS:-$OUT_ROOT/f_metrics/best_cells.json}"
GATE3_PATH="${EPM_2162_GATE3:-$OUT_ROOT/separation_gate_report.json}"
PIDFILE="$LOG_DIR/issue-2162-workers.pid"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

# Worker count = realized GPU count (gate 2: derived, never hardcoded).
NUM_WORKERS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
NUM_WORKERS="${NUM_WORKERS:-0}"
if [ "$NUM_WORKERS" -lt 1 ]; then
  NUM_WORKERS=1
fi
echo "[dispatch] phase=$PHASE num_workers=$NUM_WORKERS out_root=$OUT_ROOT"

COMMON=(--out-root "$OUT_ROOT" --log-dir "$LOG_DIR" "$@")

run_import_check() {
  echo "[phase=import-check]"
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --import-check
}

run_single_gpu_phase() {
  # bank / pilot run on ONE GPU (worker 0); rc captured through the tee pipe.
  local phase="$1"
  shift
  local log="$LOG_DIR/issue-2162-${phase}.log"
  echo "[dispatch] $phase -> $log"
  set +e
  CUDA_VISIBLE_DEVICES=0 uv run python "$DRIVER" --phase "${phase/pilot/grid}" \
    "${COMMON[@]}" "$@" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] $phase exited rc=$rc"
    exit "$rc"
  fi
}

run_fanout_phase() {
  # anchors / grid / margin: one worker per visible GPU, PLAIN backgrounded
  # children (no setsid — `wait` must be real; a detached child makes the
  # wave-chain concurrent, gotchas.md #1738). Extra args ("$@") thread in
  # BEFORE the pinned per-worker flags so the pins always win.
  local phase="$1"
  shift
  : > "$PIDFILE"
  local pids=()
  for ((g = 0; g < NUM_WORKERS; g++)); do
    local log="$LOG_DIR/issue-2162-${phase}-w${g}.log"
    echo "[dispatch] $phase worker=$g gpu=$g -> $log"
    CUDA_VISIBLE_DEVICES="$g" uv run python "$DRIVER" --phase "$phase" \
      "${COMMON[@]}" "$@" --worker-index "$g" --num-workers "$NUM_WORKERS" \
      --gpu-id "$g" > "$log" 2>&1 &
    pids+=("$!")
    echo "$!" >> "$PIDFILE"
  done
  local rc_all=0
  for ((g = 0; g < NUM_WORKERS; g++)); do
    set +e
    wait "${pids[$g]}"
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      echo "[dispatch] $phase worker=$g exited rc=$rc (log tail below)"
      tail -n 120 "$LOG_DIR/issue-2162-${phase}-w${g}.log" || true
      rc_all="$rc"
    fi
  done
  if [ "$rc_all" -ne 0 ]; then
    echo "[dispatch] $phase FAILED rc=$rc_all"
    exit "$rc_all"
  fi
}

require_gate3() {
  # r1 M4: plan §7 gate 3 (anchor separation) must PASS before the
  # 42k-rollout grid spend. The report is judge-built VM-side
  # (issue2162_judge.py --phase separation-gate) and staged to $GATE3_PATH.
  # Skip ONLY with a recorded justification:
  #   EPM_2162_SKIP_GATE3=1 EPM_2162_SKIP_GATE3_REASON="<why, >=10 chars>"
  if [ -n "${EPM_2162_SKIP_GATE3:-}" ]; then
    local reason="${EPM_2162_SKIP_GATE3_REASON:-}"
    if [ "${#reason}" -lt 10 ]; then
      echo "[dispatch] EPM_2162_SKIP_GATE3 set without a recorded justification" \
        "(EPM_2162_SKIP_GATE3_REASON, >=10 chars) — refusing" >&2
      exit 26
    fi
    echo "[dispatch] gate3 SKIPPED (recorded justification: $reason)"
    return 0
  fi
  if [ ! -f "$GATE3_PATH" ]; then
    echo "[dispatch] grid HALT rc=26: gate-3 report missing at $GATE3_PATH" \
      "(run issue2162_judge.py --phase separation-gate VM-side, stage it; r1 M4)" >&2
    exit 26
  fi
  if ! uv run python -c \
    "import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get('passed') else 1)" \
    "$GATE3_PATH"; then
    echo "[dispatch] grid HALT rc=26: gate-3 report at $GATE3_PATH is FAIL —" \
      "fix the instrument/bank per plan §7 before the grid spend" >&2
    exit 26
  fi
  echo "[dispatch] gate3 PASS ($GATE3_PATH)"
}

run_margin_if_pools() {
  # STANDALONE `margin` phase (r1 C4, scope narrowed by r2 MAJOR 1): an
  # explicitly requested margin with no pools staged is a DESIGNED HALT
  # (distinct rc, never a silent skip): stage the judge-built pools
  # (issue2162_judge.py --phase pools -> pools.json -> pod), then re-run
  # `margin` + `upload` — grid outputs are already per-worker uploaded.
  if [ ! -f "$POOLS_PATH" ]; then
    echo "[dispatch] margin HALT rc=24: pools file missing at $POOLS_PATH" \
      "(judge-built; see header). Re-run phases: margin, upload." >&2
    exit 24
  fi
  run_fanout_phase margin --pools "$POOLS_PATH"
}

run_margin_opportunistic() {
  # r2 MAJOR 1: on the `all` chain margin is OPPORTUNISTIC — the pools ride
  # the Batch-API judge SLA (2-24h), and gating upload/sentinel/teardown on
  # them idles the 8-GPU pod at 0% for the SLA tail (#664; plan §9 directive
  # 5: width released before the tail). Pools present -> margin rides the
  # wide pod now. Absent -> DEFER LOUDLY and proceed: the sentinel carries
  # margin_deferred=true + the recipe (never a silent drop), and the
  # deferred leg runs later on a fresh 1x H100 (dispatch.sh margin+upload).
  if [ ! -f "$POOLS_PATH" ]; then
    echo "[dispatch] margin DEFERRED: pools file missing at $POOLS_PATH" \
      "(judge-built; Batch-API SLA tail). Proceeding to upload + teardown;" \
      "the sentinel records margin_deferred=true + the deferred-leg recipe." \
      "Later, once pools land: dispatch.sh margin && dispatch.sh upload (1x H100)."
    return 0
  fi
  run_fanout_phase margin --pools "$POOLS_PATH"
}

run_stage2_fanout() {
  # r1 M9: stage-2 layer x dose fan-out. Own arg surface (no --log-dir /
  # --pools), so this mirrors run_fanout_phase rather than reusing COMMON.
  if [ ! -f "$BEST_CELLS_PATH" ]; then
    echo "[dispatch] stage2 HALT rc=25: best-cells file missing at" \
      "$BEST_CELLS_PATH (built by the stage-1 selection analysis)." >&2
    exit 25
  fi
  : > "$PIDFILE"
  local pids=()
  local g
  for ((g = 0; g < NUM_WORKERS; g++)); do
    local log="$LOG_DIR/issue-2162-stage2-w${g}.log"
    echo "[dispatch] stage2 worker=$g gpu=$g -> $log"
    CUDA_VISIBLE_DEVICES="$g" uv run python "$STAGE2_DRIVER" --phase stage2 \
      --out-root "$OUT_ROOT" --best-cells "$BEST_CELLS_PATH" "$@" \
      --worker-index "$g" --num-workers "$NUM_WORKERS" \
      --gpu-id "$g" > "$log" 2>&1 &
    pids+=("$!")
    echo "$!" >> "$PIDFILE"
  done
  local rc_all=0
  for ((g = 0; g < NUM_WORKERS; g++)); do
    set +e
    wait "${pids[$g]}"
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      echo "[dispatch] stage2 worker=$g exited rc=$rc (log tail below)"
      tail -n 120 "$LOG_DIR/issue-2162-stage2-w${g}.log" || true
      rc_all="$rc"
    fi
  done
  if [ "$rc_all" -ne 0 ]; then
    echo "[dispatch] stage2 FAILED rc=$rc_all"
    exit "$rc_all"
  fi
  # Stage-2 sentinel/upload phase (single process, CPU-bound). FULLY
  # redirected: the stage2 driver emits its own `[phase=done]` (it doubles
  # as a standalone dispatch), which must never reach THIS dispatcher's
  # stdout — the poller keys on the single terminal line (lint
  # --check-phase-done-reserved).
  local ulog="$LOG_DIR/issue-2162-stage2-upload.log"
  set +e
  uv run python "$STAGE2_DRIVER" --phase upload --out-root "$OUT_ROOT" \
    --best-cells "$BEST_CELLS_PATH" "$@" > "$ulog" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] stage2 upload exited rc=$rc (log tail below)"
    tail -n 120 "$ulog" || true
    exit "$rc"
  fi
  echo "[dispatch] stage2 upload complete -> $ulog"
}

run_upload() {
  echo "[phase-dispatch] upload"
  local log="$LOG_DIR/issue-2162-upload.log"
  set +e
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --phase upload \
    "${COMMON[@]}" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] upload exited rc=$rc"
    exit "$rc"
  fi
}

case "$PHASE" in
  import-check)
    run_import_check
    ;;
  bank)
    run_single_gpu_phase bank
    ;;
  anchors)
    run_fanout_phase anchors
    ;;
  pilot)
    # --num-workers threads the REALIZED width into the pilot's projection
    # (r1 C2: without it the gate computed at argparse default 1 and refused
    # any realization not ~NUM_WORKERS x faster than plan).
    run_single_gpu_phase pilot --pilot --num-workers "$NUM_WORKERS"
    ;;
  grid)
    require_gate3
    run_fanout_phase grid
    ;;
  margin)
    run_margin_if_pools
    ;;
  stage2)
    run_stage2_fanout
    ;;
  upload)
    run_upload
    ;;
  all)
    run_import_check
    run_single_gpu_phase bank
    run_fanout_phase anchors
    # r2 MINOR 1: pilot BEFORE the gate-3 report check — the pilot is
    # gate-independent (~8-15 min), so a judge/scp lag on the gate-3 report
    # is absorbed by useful work instead of halting 8 idle GPUs; a genuine
    # gate-3 FAIL forfeits only the pilot's ~180 rollouts.
    run_single_gpu_phase pilot --pilot --num-workers "$NUM_WORKERS"
    require_gate3
    run_fanout_phase grid
    run_margin_opportunistic
    run_upload
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

echo "[phase=done]"
