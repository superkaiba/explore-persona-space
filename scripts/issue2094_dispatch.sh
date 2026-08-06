#!/usr/bin/env bash
# Issue #2094 dispatcher — per-GPU worker fan-out for the single-position
# context/prefix intervention grid (plan v4 §4.6 / §9 / §10).
#
# Phases, in order:
#   bank    (P1) worker 0 only — V bank + injection-exactness gate (rc 21 halts)
#   anchors (P2) worker 0 only — 15 contexts x K=10 unpatched draws + V_a
#   pilot   (P3 entry) worker 0 only — ONE production-shape block family timed
#                                      through the production entrypoint (rc 22 halts)
#   grid    (P3+P4) N workers — 880 blocks sharded round-robin, one GPU each
#   upload  (P5) worker 0 only — ONE bulk commit per HF prefix + the pod sentinel
#
# CVD discipline (the #545 import-time-cuInit family): CUDA_VISIBLE_DEVICES is
# set PER WORKER in the LAUNCHER ENV — never exported globally — and the matching
# --gpu-id rides along so the in-process pin (if any) rewrites the SAME value.
#
# Pod-side contract: `[phase=...]` breadcrumbs + the sentinel the DRIVER writes.
# This script NEVER shells out to scripts/task.py (pods run issue-<N> branches;
# task.py branch-guards to main).
#
# Python-first error routing: every failure arm exits with the DRIVER's rc — no
# bare `false` inside a compound branch (the embedded-shell exit-path rule).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"
# Conditional .env sourcing — the GCE lane has NO .env (metadata exports instead).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

PHASE="${1:-all}"
if [ "$PHASE" = "--phase" ]; then PHASE="${2:-all}"; shift 2 || true; fi

NUM_WORKERS="${EPM_2094_NUM_WORKERS:-}"
if [ -z "$NUM_WORKERS" ]; then
  # Realized width, never a hardcoded 8: the dispatcher re-shards off what the
  # provision actually gave us (plan §9 "dispatcher re-shards off realized width").
  NUM_WORKERS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  NUM_WORKERS="${NUM_WORKERS:-0}"
  if [ "$NUM_WORKERS" -lt 1 ]; then NUM_WORKERS=1; fi
fi

OUT_ROOT="${EPM_2094_OUT_ROOT:-/workspace/issue2094_out}"
LOG_DIR="${EPM_2094_LOG_DIR:-/workspace/logs}"
GEN_BATCH="${EPM_2094_GEN_BATCH:-16}"
CAPTURE_BATCH="${EPM_2094_CAPTURE_BATCH:-8}"
MAX_NEW_TOKENS="${EPM_2094_MAX_NEW_TOKENS:-1024}"
PLANNED_WALL_H="${EPM_2094_PLANNED_WALL_H:-3.0}"
UPLOAD_MODE="${EPM_2094_UPLOAD:-hf}"
UPLOAD_EVERY="${EPM_2094_UPLOAD_EVERY:-25}"
SMOKE_FLAG=""
if [ -n "${EPM_2094_SMOKE:-}" ]; then SMOKE_FLAG="--smoke"; fi
EXTRA_FLAGS="${EPM_2094_EXTRA_FLAGS:-}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
DRIVER="scripts/issue2094_run.py"
PIDFILE="$LOG_DIR/issue-2094.pid"

COMMON=(
  --out-root "$OUT_ROOT"
  --log-dir "$LOG_DIR"
  --gen-batch "$GEN_BATCH"
  --capture-batch "$CAPTURE_BATCH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --planned-wall-h "$PLANNED_WALL_H"
  --upload "$UPLOAD_MODE"
  --upload-every "$UPLOAD_EVERY"
)
if [ -n "$SMOKE_FLAG" ]; then COMMON+=("$SMOKE_FLAG"); fi
# shellcheck disable=SC2206  # deliberate word-splitting of an operator-supplied flag string
if [ -n "$EXTRA_FLAGS" ]; then COMMON+=($EXTRA_FLAGS); fi

echo "[dispatch] phase=$PHASE workers=$NUM_WORKERS out_root=$OUT_ROOT smoke=${EPM_2094_SMOKE:-0}"
echo "$$" > "$PIDFILE"

run_import_check() {
  echo "[phase=import_check]"
  # Resolves every DEFERRED import the driver reaches on its REAL paths
  # (transformers / huggingface_hub / hub upload helpers live inside function
  # bodies), which a bare `import <module>` never fires (#1689).
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --import-check
}

run_bank() {
  echo "[phase=bank_launch] gpu=0"
  set +e
  CUDA_VISIBLE_DEVICES=0 uv run python "$DRIVER" --phase bank --gpu-id 0 \
    "${COMMON[@]}" 2>&1 | tee "$LOG_DIR/issue-2094-bank.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[phase=bank_failed] rc=$rc report=$OUT_ROOT/vc_bank/injection_gate_report.json"
    exit "$rc"
  fi
}

run_anchors() {
  echo "[phase=anchors_launch] gpu=0"
  set +e
  CUDA_VISIBLE_DEVICES=0 uv run python "$DRIVER" --phase anchors --gpu-id 0 \
    "${COMMON[@]}" 2>&1 | tee "$LOG_DIR/issue-2094-anchors.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[phase=anchors_failed] rc=$rc"
    exit "$rc"
  fi
}

run_pilot() {
  echo "[phase=pilot_launch] gpu=0 width=$NUM_WORKERS"
  set +e
  CUDA_VISIBLE_DEVICES=0 uv run python "$DRIVER" --phase grid --pilot \
    --worker-index 0 --num-workers "$NUM_WORKERS" --gpu-id 0 \
    "${COMMON[@]}" 2>&1 | tee "$LOG_DIR/issue-2094-pilot.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[phase=pilot_failed] rc=$rc report=$OUT_ROOT/pilot_gate_report.json"
    exit "$rc"
  fi
}

run_grid() {
  echo "[phase=grid_fanout] width=$NUM_WORKERS"
  # PLAIN backgrounded children (no setsid) so `wait` below is real — a
  # setsid-detached shard reparents to pid 1 and `wait` returns instantly
  # (the #1738 chained-waves trap). The LAUNCHER is what runs detached.
  PIDS=()
  for g in $(seq 0 $((NUM_WORKERS - 1))); do
    CUDA_VISIBLE_DEVICES="$g" nohup uv run python "$DRIVER" --phase grid \
      --worker-index "$g" --num-workers "$NUM_WORKERS" --gpu-id "$g" \
      "${COMMON[@]}" > "$LOG_DIR/issue-2094-grid-w$g.log" 2>&1 < /dev/null &
    PIDS+=($!)
    echo "[grid] worker $g pid=${PIDS[$g]} log=$LOG_DIR/issue-2094-grid-w$g.log"
  done
  echo "$$ ${PIDS[*]}" > "$PIDFILE"
  FAIL=0
  FAIL_RC=1
  for g in $(seq 0 $((NUM_WORKERS - 1))); do
    set +e
    wait "${PIDS[$g]}"
    wrc=$?
    set -e
    if [ "$wrc" -ne 0 ]; then
      echo "[phase=grid_worker_failed] worker=$g rc=$wrc"
      FAIL=1
      FAIL_RC="$wrc"
    fi
  done
  if [ "$FAIL" -ne 0 ]; then
    echo "[phase=grid_failed] rc=$FAIL_RC — see per-worker logs under $LOG_DIR"
    exit "$FAIL_RC"
  fi
  echo "$$" > "$PIDFILE"
}

run_upload() {
  echo "[phase=upload_launch]"
  # CPU-only phase (HF upload_folder uses no GPU) — CVD emptied so it can never
  # squat a device the next phase needs.
  set +e
  CUDA_VISIBLE_DEVICES="" uv run python "$DRIVER" --phase upload \
    "${COMMON[@]}" 2>&1 | tee "$LOG_DIR/issue-2094-upload.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[phase=upload_failed] rc=$rc"
    exit "$rc"
  fi
}

case "$PHASE" in
  import-check) run_import_check ;;
  bank)         run_import_check; run_bank ;;
  anchors)      run_anchors ;;
  pilot)        run_pilot ;;
  grid)         run_grid ;;
  upload)       run_upload ;;
  all)
    run_import_check
    run_bank
    run_anchors
    run_pilot
    run_grid
    run_upload
    ;;
  *)
    echo "[dispatch] unknown phase: $PHASE (expected: import-check|bank|anchors|pilot|grid|upload|all)" >&2
    exit 2
    ;;
esac

echo "[phase=done]"
