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
#   upload          P5: bulk HF uploads + pod sentinel (CPU)
#   all             import-check -> bank -> anchors -> pilot -> grid -> upload
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
OUT_ROOT="${EPM_2162_OUT_ROOT:-/workspace/issue2162_out}"
LOG_DIR="${EPM_2162_LOG_DIR:-/workspace/logs}"
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
  # wave-chain concurrent, gotchas.md #1738).
  local phase="$1"
  shift
  : > "$PIDFILE"
  local pids=()
  for ((g = 0; g < NUM_WORKERS; g++)); do
    local log="$LOG_DIR/issue-2162-${phase}-w${g}.log"
    echo "[dispatch] $phase worker=$g gpu=$g -> $log"
    CUDA_VISIBLE_DEVICES="$g" uv run python "$DRIVER" --phase "$phase" \
      "${COMMON[@]}" --worker-index "$g" --num-workers "$NUM_WORKERS" \
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
    run_single_gpu_phase pilot --pilot
    ;;
  grid)
    run_fanout_phase grid
    ;;
  margin)
    run_fanout_phase margin
    ;;
  upload)
    run_upload
    ;;
  all)
    run_import_check
    run_single_gpu_phase bank
    run_fanout_phase anchors
    run_single_gpu_phase pilot --pilot
    run_fanout_phase grid
    run_upload
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

echo "[phase=done]"
