#!/usr/bin/env bash
# Issue #2329 q35_ladder_decay — pod dispatcher for scripts/issue2329_ladder.py
# (Leg A: the #2162 persona-specificity ladder ported to Qwen/Qwen3.5-9B).
#
# Forked from scripts/issue2329_dispatch.sh (gate-0b venv-pin pattern; plan v8
# §5 embedded-shell exit-path trace: every failure arm ends with `exit <rc>`).
# Phases (pod lanes L1/L2/L4; the driver's own G0 tokgate phase is a VM
# deliverable and is NOT dispatched here):
#   gate0b        plan gate 0b: pin transformers==5.15.0 into the pod venv,
#                 then run the shared rig's asserts (scripts/issue2329_run.py
#                 --gate0b-check: version + AutoConfig qwen3_5 + 32 blocks; CPU)
#   import-check  ladder-driver deferred-import resolution + the
#                 argparse-attribute completeness assert (CPU, no GPU)
#   bank          L1: parent-bank stage + ladder_bank.json freeze + all-layer
#                 v_ce/v_pe capture + G1 donor-identity gate (driver rc=24 on
#                 RC_DONOR_IDENTITY; single GPU)
#   anchors       L2: 42 contexts x K=10 unpatched rollouts; the FIRST claim
#                 block is the measured throughput pilot (G2, driver rc=28 on
#                 RC_THROUGHPUT_GATE); fanned out per visible GPU
#   grid          L4: gate-surviving (direction x slot x arm) claim-queue
#                 blocks. HARD-requires the three staged verdict files below
#                 (all built OFF-POD during the L3 VM judge window) before any
#                 spend; the driver re-validates each (fail-loud)
#   margin        L4: pools-dependent margin TF legs (pools_ladder.json is a
#                 zero-API L3 re-reduction — staged before this launch; a
#                 missing pools file is a DESIGNED HALT rc=29, never a skip)
#   upload        L4: bulk HF upload + pod sentinel (CPU)
#   stage1        gate0b -> import-check -> bank -> anchors  (the L1+L2 launch;
#                 the pod then idles through the VM L3 judge-gate window)
#   stage2        grid-gates -> grid -> margin -> upload     (the L4 launch)
#   all           stage1 -> stage2 (tiny/smoke e2e; production uses the two
#                 stage launches so the L3 handoff stays explicit)
#
# Staged verdict files consumed by `grid` AND `margin` (defaults under
# $OUT_ROOT/gates/):
#   token_identity_report_ladder.json  G0 (VM tokgate, committed + staged)
#   ladder_separation_gate.json        G3 (judge-built off-pod, L3)
#   ladder_donor_screen.json           donor screen + pe-viability (L3)
# The gate files are required + threaded in BOTH modes — --smoke included
# (review round 1 must-fix 8: no gate downgrades; the driver slices the
# gate-threaded production enumeration down to the 12-cell smoke subset via
# smoke_slice_blocks, so the smoke exercises the real gate inputs).
#
# Worker count is DERIVED from the realized GPU count (`nvidia-smi -L`) at
# launch — never hardcoded (pod-2329-l is 1x H100, so the fan-out realizes
# width 1; the shape stays width-correct on any wider provision). Each worker
# gets CUDA_VISIBLE_DEVICES pinned in ITS OWN launcher env (never `+gpu_id=N`;
# the in-process clobber is defeated by import-time cuInit — gotchas.md).
#
# UV_NO_SYNC=1 for the whole dispatch: gate0b pins transformers==5.15.0 with
# `uv pip install`, and a bare `uv run` would re-sync to uv.lock (4.57.6) and
# silently UNDO the pin; an N-way `uv run` fan-out re-resolving on the MooseFS
# venv is also the #1689 FUSE read-wedge trigger.
#
# The single terminal `[phase=done]` line is emitted ONLY at the very end of a
# successful dispatch invocation (the poller keys on it).

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

export UV_NO_SYNC=1

PHASE="${1:-all}"
shift || true

DRIVER="scripts/issue2329_ladder.py"
GATE0B_DRIVER="scripts/issue2329_run.py"
OUT_ROOT="${EPM_2329L_OUT_ROOT:-/workspace/issue2329_out/ladder}"
LOG_DIR="${EPM_2329L_LOG_DIR:-/workspace/logs}"
TOKEN_IDENTITY="${EPM_2329L_TOKEN_IDENTITY:-$OUT_ROOT/gates/token_identity_report_ladder.json}"
GATE_VERDICT="${EPM_2329L_GATE_VERDICT:-$OUT_ROOT/gates/ladder_separation_gate.json}"
DONOR_SCREEN="${EPM_2329L_DONOR_SCREEN:-$OUT_ROOT/gates/ladder_donor_screen.json}"
POOLS_PATH="${EPM_2329L_POOLS:-$OUT_ROOT/pools_ladder.json}"
TRANSFORMERS_PIN="${EPM_2329L_TRANSFORMERS_PIN:-5.15.0}"
PIDFILE="$LOG_DIR/issue-2329-ladder-workers.pid"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

# Worker count = realized GPU count (derived, never hardcoded). Plan §9 pins
# backend: runpod (pod-2329-l, dedicated pod — never a shared SLURM node), so
# dedicated-pod nvidia-smi enumeration IS the allocation width.
# SLURM_GPU_WIDTH_EXEMPT: runpod-pinned dedicated-pod dispatcher (plan §9)
NUM_WORKERS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
NUM_WORKERS="${NUM_WORKERS:-0}"
if [ "$NUM_WORKERS" -lt 1 ]; then
  NUM_WORKERS=1
fi
echo "[dispatch] phase=$PHASE num_workers=$NUM_WORKERS out_root=$OUT_ROOT"

COMMON=(--out-root "$OUT_ROOT" --log-dir "$LOG_DIR" "$@")

run_gate0b() {
  # Plan gate 0b: qwen3_5 needs transformers==5.15.0 (repo pin 4.57.6 lacks
  # it). Pin the pod venv FIRST, then run the shared rig's asserts on CPU.
  echo "[phase=gate0b]"
  local log="$LOG_DIR/issue-2329-ladder-gate0b.log"
  set +e
  uv pip install "transformers==${TRANSFORMERS_PIN}" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] gate0b pip install exited rc=$rc"
    exit "$rc"
  fi
  set +e
  CUDA_VISIBLE_DEVICES="" uv run python "$GATE0B_DRIVER" --gate0b-check 2>&1 | tee -a "$log"
  rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] gate0b exited rc=$rc"
    exit "$rc"
  fi
}

run_import_check() {
  echo "[phase=import-check]"
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --import-check
}

run_single_gpu_phase() {
  # bank runs on ONE GPU (worker 0); rc captured through the tee pipe.
  local phase="$1"
  shift
  local log="$LOG_DIR/issue-2329-ladder-${phase}.log"
  echo "[dispatch] $phase -> $log"
  set +e
  CUDA_VISIBLE_DEVICES=0 uv run python "$DRIVER" --phase "$phase" \
    "${COMMON[@]}" "$@" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[dispatch] $phase exited rc=$rc"
    exit "$rc"
  fi
}

run_cpu_phase() {
  # upload: single CPU process, no GPU pin.
  local phase="$1"
  shift
  local log="$LOG_DIR/issue-2329-ladder-${phase}.log"
  echo "[dispatch] $phase -> $log"
  set +e
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --phase "$phase" \
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
  local g
  for ((g = 0; g < NUM_WORKERS; g++)); do
    local log="$LOG_DIR/issue-2329-ladder-${phase}-w${g}.log"
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
      tail -n 120 "$LOG_DIR/issue-2329-ladder-${phase}-w${g}.log" || true
      rc_all="$rc"
    fi
  done
  if [ "$rc_all" -ne 0 ]; then
    echo "[dispatch] $phase FAILED rc=$rc_all"
    exit "$rc_all"
  fi
}

require_grid_gates() {
  # The three staged verdict files the grid spend HARD-requires (plan §4.5;
  # all built off-pod: G0 on the VM at L0, G3 + donor screen at L3). The
  # driver re-validates each file's CONTENT (read_gate_verdict /
  # read_donor_screen / token-identity consumption fail loud); this check
  # fails FAST, before any worker fan-out. Skip ONLY with a recorded
  # justification: EPM_2329L_SKIP_GRID_GATES=1
  # EPM_2329L_SKIP_GRID_GATES_REASON="<why, >=10 chars>".
  # Explicit "1" compare (g5-2): any other truthy-looking value (e.g. "0",
  # "false") must NOT skip the gates.
  if [ "${EPM_2329L_SKIP_GRID_GATES:-}" = "1" ]; then
    local reason="${EPM_2329L_SKIP_GRID_GATES_REASON:-}"
    if [ "${#reason}" -lt 10 ]; then
      echo "[dispatch] EPM_2329L_SKIP_GRID_GATES set without a recorded justification" \
        "(EPM_2329L_SKIP_GRID_GATES_REASON, >=10 chars) — refusing" >&2
      exit 26
    fi
    echo "[dispatch] grid gates SKIPPED (recorded justification: $reason)"
    return 0
  fi
  if [ ! -f "$TOKEN_IDENTITY" ]; then
    echo "[dispatch] grid HALT rc=30: G0 token-identity report missing at" \
      "$TOKEN_IDENTITY (VM tokgate deliverable; stage it, then re-run)" >&2
    exit 30
  fi
  if [ ! -f "$GATE_VERDICT" ]; then
    echo "[dispatch] grid HALT rc=26: anchor-separation gate verdict missing at" \
      "$GATE_VERDICT (judge-built off-pod at L3; stage it, then re-run)" >&2
    exit 26
  fi
  if ! uv run python -c \
    "import json, sys; r = json.load(open(sys.argv[1])).get('rungs') or {}; raise SystemExit(0 if any(v.get('survived') for v in r.values()) else 1)" \
    "$GATE_VERDICT"; then
    echo "[dispatch] grid HALT rc=26: gate verdict at $GATE_VERDICT has NO surviving" \
      "rungs — the ALL-rungs-fail HALT is a rig-defect branch, not a grid input" >&2
    exit 26
  fi
  if [ ! -f "$DONOR_SCREEN" ]; then
    echo "[dispatch] grid HALT rc=25: donor screen missing at $DONOR_SCREEN" \
      "(judge-built off-pod at L3; stage it, then re-run)" >&2
    exit 25
  fi
  echo "[dispatch] grid gates staged: $TOKEN_IDENTITY | $GATE_VERDICT | $DONOR_SCREEN"
}

run_grid() {
  # Gates are hard-required + threaded in BOTH modes (must-fix 8: --smoke no
  # longer bypasses the three production grid-gate inputs; the driver's
  # smoke_slice_blocks slices the gate-threaded enumeration to the smoke
  # subset).
  require_grid_gates
  run_fanout_phase grid \
    --token-identity "$TOKEN_IDENTITY" \
    --gate-verdict "$GATE_VERDICT" \
    --donor-screen "$DONOR_SCREEN"
}

run_margin() {
  # phase_margin re-enumerates blocks via the driver's _grid_inputs, which
  # hard-requires the same three gate files as grid — require + thread them
  # here too (a margin launch without them would rc-crash inside the driver).
  require_grid_gates
  # pools_ladder.json is a zero-API L3 re-reduction staged before the L4
  # launch; a missing file on an explicitly requested margin is a DESIGNED
  # HALT (distinct rc, never a silent skip).
  if [ ! -f "$POOLS_PATH" ]; then
    echo "[dispatch] margin HALT rc=29: pools file missing at $POOLS_PATH" \
      "(built by scripts/issue2329_ladder_judge.py --phase pools at L3;" \
      "stage it, then re-run: margin, upload)." >&2
    exit 29
  fi
  run_fanout_phase margin --pools "$POOLS_PATH" \
    --token-identity "$TOKEN_IDENTITY" \
    --gate-verdict "$GATE_VERDICT" \
    --donor-screen "$DONOR_SCREEN"
}

run_upload() {
  # Thread the realized width so the sentinel's recorded num_workers stays
  # truthful (the driver derives each family's width from its OWN done-record
  # set and never sweeps on this value).
  run_cpu_phase upload --num-workers "$NUM_WORKERS"
}

run_stage1() {
  run_gate0b
  run_import_check
  run_single_gpu_phase bank
  run_fanout_phase anchors
}

run_stage2() {
  # Re-assert the venv pin (g5-1): stage2 is a SEPARATE dispatch invocation —
  # a pod resume / uv re-sync between stage1 and stage2 can silently revert
  # transformers to the uv.lock pin (4.57.6), and gate0b is idempotent + fast.
  run_gate0b
  run_grid
  run_margin
  run_upload
}

case "$PHASE" in
  gate0b)
    run_gate0b
    ;;
  import-check)
    run_import_check
    ;;
  bank)
    run_single_gpu_phase bank
    ;;
  anchors)
    run_fanout_phase anchors
    ;;
  grid)
    run_grid
    ;;
  margin)
    run_margin
    ;;
  upload)
    run_upload
    ;;
  stage1)
    run_stage1
    ;;
  stage2)
    run_stage2
    ;;
  all)
    run_stage1
    run_stage2
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

echo "[phase=done]"
