#!/usr/bin/env bash
# Issue #2162 turn-boundary-multipatch (tbmp) — pod dispatcher for
# scripts/issue2162_tbmp.py. Forked from scripts/issue2162_dispatch.sh
# (same worker-count derivation, per-worker CVD pins, claim-queue fan-out,
# single terminal `[phase=done]`).
#
# Phases:
#   import-check    deferred-import resolution + argparse-attribute
#                   completeness (CPU, no GPU)
#   bank            P1: boundary resolver + G1 gate (incl. assumption-9
#                   rebuilt-vs-recorded context parity + G1(b) len_delta
#                   parity) + tb bank capture + multi-position injection
#                   gate (GPU 0; rc=24 G1 HALT, rc=21 injection HALT —
#                   driver-owned designed halts; rc=28 dispatcher HALT
#                   when the parent bank.json is not staged)
#   pilot           P2-entry: ONE production-shape timed tb block (GPU 0;
#                   rc=22 refusal via the shared parent pilot gate)
#   grid            P2: 45-block claim-file queue, one worker per visible GPU
#                   (margins inline when the pools file is staged)
#   margin          pools-dependent margin TF catch-up, one worker per GPU
#                   (rc=27 HALT when pools are missing — an explicitly
#                   requested margin with no pools is an error)
#   upload          P3: bulk HF uploads + pod sentinel (CPU)
#   all             import-check -> bank -> pilot -> grid
#                   -> margin (OPPORTUNISTIC) -> upload
#
# Margin semantics inherit the parent's r2 MAJOR 1 shape: the pools file is
# judge-built (Batch-API SLA tail), so the `all` chain never parks the wide
# pod behind it — pools staged -> margins ride the grid inline / the margin
# leg here; absent -> DEFER LOUDLY (sentinel carries margin_deferred=true +
# the recipe; the deferred leg runs later on a 1x H100).
#
# Worker count is DERIVED from the realized GPU count at launch — never
# hardcoded; each worker gets CUDA_VISIBLE_DEVICES pinned in ITS OWN
# launcher env (the in-process clobber is defeated by import-time cuInit —
# gotchas.md). The single terminal `[phase=done]` line is emitted ONLY at
# the very end of a successful dispatch (the poller keys on it).

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

DRIVER="scripts/issue2162_tbmp.py"
OUT_ROOT="${EPM_2162_TBMP_OUT_ROOT:-/workspace/issue2162_out/tbmp}"
LOG_DIR="${EPM_2162_TBMP_LOG_DIR:-/workspace/logs}"
POOLS_PATH="${EPM_2162_TBMP_POOLS:-$OUT_ROOT/pools.json}"
PARENT_BANK_PATH="${EPM_2162_TBMP_PARENT_BANK:-$OUT_ROOT/parent_bank.json}"
PIDFILE="$LOG_DIR/issue-2162-tbmp-workers.pid"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

# Worker count = realized GPU count (derived, never hardcoded).
# SLURM_GPU_WIDTH_EXEMPT: RunPod pod-only dispatcher (pod.py provision; sentinel lane) — never dispatched on SLURM lanes
NUM_WORKERS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
NUM_WORKERS="${NUM_WORKERS:-0}"
if [ "$NUM_WORKERS" -lt 1 ]; then
  NUM_WORKERS=1
fi
echo "[dispatch] phase=$PHASE num_workers=$NUM_WORKERS out_root=$OUT_ROOT"

COMMON=(--out-root "$OUT_ROOT" --log-dir "$LOG_DIR" "$@")

# MANDATORY parity input for the bank phase (plan §4.5 DAG stages bank.json
# unconditionally at P1): shuffled-assignment + assumption-9 context parity.
# A missing bank is a DESIGNED HALT (rc=28 — distinct from rc=24 G1 /
# rc=27 margin-pools), never a silent recompute-only run.
BANK_ARGS=()
require_parent_bank() {
  if [ ! -f "$PARENT_BANK_PATH" ]; then
    echo "[dispatch] bank HALT rc=28: parent bank missing at $PARENT_BANK_PATH" \
      "(stage issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json from HF, or set" \
      "EPM_2162_TBMP_PARENT_BANK). Re-run: bank (or all)." >&2
    exit 28
  fi
  BANK_ARGS=(--parent-bank "$PARENT_BANK_PATH")
  echo "[dispatch] parent bank staged: $PARENT_BANK_PATH (parity checks armed)"
}

run_import_check() {
  echo "[phase=import-check]"
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --import-check
}

run_single_gpu_phase() {
  # bank / pilot run on ONE GPU (worker 0); rc captured through the tee pipe.
  local phase="$1"
  shift
  local log="$LOG_DIR/issue-2162-tbmp-${phase}.log"
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
  # grid / margin: one worker per visible GPU, PLAIN backgrounded children
  # (no setsid — `wait` must be real; a detached child makes the wave-chain
  # concurrent, gotchas.md #1738). Extra args ("$@") thread in BEFORE the
  # pinned per-worker flags so the pins always win.
  local phase="$1"
  shift
  : > "$PIDFILE"
  local pids=()
  for ((g = 0; g < NUM_WORKERS; g++)); do
    local log="$LOG_DIR/issue-2162-tbmp-${phase}-w${g}.log"
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
      tail -n 120 "$LOG_DIR/issue-2162-tbmp-${phase}-w${g}.log" || true
      rc_all="$rc"
    fi
  done
  if [ "$rc_all" -ne 0 ]; then
    echo "[dispatch] $phase FAILED rc=$rc_all"
    exit "$rc_all"
  fi
}

run_margin_if_pools() {
  # STANDALONE `margin` phase: an explicitly requested margin with no pools
  # staged is a DESIGNED HALT (rc=27 — distinct from the driver's rc=24 G1
  # halt), never a silent skip. Stage the parent's judge-built pools
  # (eval_results/issue_2162/judge/pools.json -> $POOLS_PATH), then re-run
  # `margin` + `upload` — grid outputs are already per-worker uploaded.
  if [ ! -f "$POOLS_PATH" ]; then
    echo "[dispatch] margin HALT rc=27: pools file missing at $POOLS_PATH" \
      "(stage eval_results/issue_2162/judge/pools.json). Re-run: margin, upload." >&2
    exit 27
  fi
  run_fanout_phase margin --pools "$POOLS_PATH"
}

run_margin_opportunistic() {
  # `all` chain: margin is OPPORTUNISTIC — pools present -> margin rides the
  # wide pod now; absent -> DEFER LOUDLY and proceed (the sentinel carries
  # margin_deferred=true + the deferred-leg recipe; #664 idle-burn).
  if [ ! -f "$POOLS_PATH" ]; then
    echo "[dispatch] margin DEFERRED: pools file missing at $POOLS_PATH" \
      "(judge-built; Batch-API SLA tail). Proceeding to upload + teardown;" \
      "the sentinel records margin_deferred=true + the deferred-leg recipe." \
      "Later, once pools land: tbmp_dispatch.sh margin && tbmp_dispatch.sh upload (1x H100)."
    return 0
  fi
  run_fanout_phase margin --pools "$POOLS_PATH"
}

run_grid() {
  # Margins ride the grid inline when pools are staged (driver checks
  # existence itself and defers loudly otherwise).
  run_fanout_phase grid --pools "$POOLS_PATH"
}

run_upload() {
  echo "[phase-dispatch] upload"
  local log="$LOG_DIR/issue-2162-tbmp-upload.log"
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
    require_parent_bank
    run_single_gpu_phase bank "${BANK_ARGS[@]}"
    ;;
  pilot)
    # --num-workers threads the REALIZED width into the pilot's projection
    # (parent r1 C2: without it the gate computed at argparse default 1 and
    # refused any realization not ~NUM_WORKERS x faster than plan).
    run_single_gpu_phase pilot --pilot --num-workers "$NUM_WORKERS"
    ;;
  grid)
    run_grid
    ;;
  margin)
    run_margin_if_pools
    ;;
  upload)
    run_upload
    ;;
  all)
    require_parent_bank
    run_import_check
    run_single_gpu_phase bank "${BANK_ARGS[@]}"
    run_single_gpu_phase pilot --pilot --num-workers "$NUM_WORKERS"
    run_grid
    run_margin_opportunistic
    run_upload
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

echo "[phase=done]"
