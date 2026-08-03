#!/usr/bin/env bash
# #1482 dense-context -> full-width-SAE fit grid: work-conserving per-GPU fan-out.
#
# Copied to the compute box as /workspace/launch_issue_1482_densesae.sh and started
# DETACHED (setsid nohup bash <launcher> > <log> 2>&1 < /dev/null &). The launcher
# itself owns the pidfile and waits on its per-GPU workers, so a caller can never
# chain waves against a launcher that already exited (#1738).
#
# Fan-out is work-conserving: every GPU pulls the next cell off a flock'd queue, so
# no GPU idles behind a wave barrier while an independent cell is pending. Each cell
# runs with CUDA_VISIBLE_DEVICES pinned in the WORKER ENV (the in-process clobber is
# defeated by any import-time cuInit) plus the matching --gpu-id.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-/workspace/explore-persona-space}}"

# Scratch root: /workspace exists on RunPod/GCE and on the fellows VAST mount, but
# NOT on the DRAC/Mila SLURM lanes (where $SCRATCH is the writable root). Resolve
# rather than assume, so an auto-chain fall-through cannot die at the first mkdir.
resolve_scratch() {
  for c in "${EPS_SCRATCH_DIR:-}" /workspace "${SCRATCH:-}" "${TMPDIR:-/tmp}"; do
    [ -n "$c" ] || continue
    if mkdir -p "$c/issue1482_densesae" 2>/dev/null; then echo "$c"; return; fi
  done
  echo "/tmp"
}
SCRATCH_ROOT="${SCRATCH_ROOT:-$(resolve_scratch)}"
WORK="${WORK:-$SCRATCH_ROOT/issue1482_densesae}"
OUT="${OUT:-$REPO_ROOT/eval_results/issue_1482/densesae_fullwidth}"
LOGDIR="${LOGDIR:-$SCRATCH_ROOT/logs/issue1482_densesae}"
PIDFILE="${PIDFILE:-$SCRATCH_ROOT/logs/issue-1482-densesae.pid}"
PILOT_CELL="${PILOT_CELL:-ridge__mean}"
# Cells the grid runs after the pilot. All are REQUIRED — any failure fails the
# grid (the width-32768 capacity cell was removed by user directive, so there is
# no optional tier left to tolerate).
CELLS="${CELLS:-ridge__max ridge__frac mlp__mean mlp__max mlp__frac mlpgate__mean}"
PY="${PY:-uv run python}"
DRIVER="$REPO_ROOT/scripts/issue1482_densesae_fullwidth.py"

mkdir -p "$LOGDIR" "$WORK" "$(dirname "$PIDFILE")"
echo $$ > "$PIDFILE"
cd "$REPO_ROOT" || exit 1

# The GCE lane exports its tokens into the environment and ships no .env file.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ── GPU width: from the ALLOCATION on SLURM, never device enumeration ──────────
# Fellows/SLURM nodes are GPU-SHARED and nvidia-smi always enumerates all 8
# physical devices regardless of the allocation, so a detected-count fan-out would
# shard onto other tenants' GPUs.
resolve_gpus() {
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
      echo "${CUDA_VISIBLE_DEVICES//,/ }"; return
    fi
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then echo "${SLURM_JOB_GPUS//,/ }"; return; fi
    if [ -n "${SLURM_STEP_GPUS:-}" ]; then echo "${SLURM_STEP_GPUS//,/ }"; return; fi
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
      seq 0 $((SLURM_GPUS_ON_NODE - 1)) | tr '\n' ' '; return
    fi
    log "FATAL: SLURM job exposes no GPU allocation env; refusing to guess width"
    exit 2
  fi
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "${CUDA_VISIBLE_DEVICES//,/ }"; return; fi
  nvidia-smi -L 2>/dev/null | awk -F: '/^GPU /{print $1}' | awk '{print $2}' | tr '\n' ' '
}

GPUS=($(resolve_gpus))
NGPU=${#GPUS[@]}
if [ "$NGPU" -lt 1 ]; then log "FATAL: no GPUs resolved"; exit 2; fi
log "[phase=start] gpus=(${GPUS[*]}) ngpu=$NGPU work=$WORK out=$OUT"

run_cell() {  # $1=cell  $2=gpu  $3..=extra driver args
  local cell="$1" gpu="$2" rc=0
  shift 2
  log "[phase=cell_start] cell=$cell gpu=$gpu extra='$*'"
  CUDA_VISIBLE_DEVICES="$gpu" $PY "$DRIVER" --phase fit --device cuda --gpu-id "$gpu" \
    --cells "$cell" --work "$WORK" --out "$OUT" "$@" \
    >> "$LOGDIR/cell_${cell}.log" 2>&1 || rc=$?
  log "[phase=cell_done] cell=$cell gpu=$gpu rc=$rc"
  return $rc
}

# ── serial prologue: stage + assemble (shared by every cell) ──────────────────
if [ "${SKIP_STAGE:-0}" != "1" ]; then
  log "[phase=stage]"
  $PY "$DRIVER" --phase stage --work "$WORK" --out "$OUT" >> "$LOGDIR/stage.log" 2>&1 || {
    log "FATAL: stage failed (see $LOGDIR/stage.log)"; exit 3; }
fi
log "[phase=assemble]"
$PY "$DRIVER" --phase assemble --work "$WORK" --out "$OUT" >> "$LOGDIR/assemble.log" 2>&1 || {
  log "FATAL: assemble failed (see $LOGDIR/assemble.log)"; exit 4; }

# ── MEASURED 1-cell pilot on GPU 0, at PRODUCTION shape, through this same path ──
if [ "${SKIP_PILOT:-0}" != "1" ]; then
  # --verify-xty rides the PILOT only: the cuSPARSE X^T Y branch is CUDA-only, so
  # no CPU-host smoke can reach it, and this is its fix-engaged signal. The probe
  # is row-subset-scoped, so it costs seconds and does not distort the measured
  # per-cell wall the grid is sized from.
  log "[phase=pilot] cell=$PILOT_CELL"
  t0=$(date +%s)
  run_cell "$PILOT_CELL" "${GPUS[0]}" --verify-xty || {
    log "FATAL: pilot cell failed"; exit 5; }
  t1=$(date +%s)
  echo "{\"pilot_cell\": \"$PILOT_CELL\", \"wall_s\": $((t1 - t0))}" > "$WORK/pilot_wall.json"
  log "[phase=pilot_done] cell=$PILOT_CELL wall_s=$((t1 - t0))"
fi

# ── work-conserving grid: each GPU pulls the next pending cell ─────────────────
QUEUE="$WORK/cell_queue.txt"
LOCK="$WORK/cell_queue.lock"
FAILED="$WORK/failed_cells.txt"
: > "$FAILED"; : > "$LOCK"
printf '%s\n' $CELLS > "$QUEUE"
log "[phase=grid] queued: $(tr '\n' ' ' < "$QUEUE")"

worker() {
  local gpu="$1" cell
  while :; do
    cell=$(flock "$LOCK" -c "head -n1 '$QUEUE'; sed -i '1d' '$QUEUE'")
    [ -z "$cell" ] && break
    if ! run_cell "$cell" "$gpu"; then
      flock "$LOCK" -c "echo '$cell' >> '$FAILED'"
    fi
  done
  log "[phase=worker_done] gpu=$gpu"
}

for g in "${GPUS[@]}"; do worker "$g" & done
wait

# One summary AFTER joining every worker: each cell process writes its own on
# exit, so without this the last finisher decides what the grid summary contains.
$PY "$DRIVER" --phase summary --work "$WORK" --out "$OUT" >> "$LOGDIR/summary.log" 2>&1 || \
  log "WARN: summary rebuild failed (see $LOGDIR/summary.log)"

# ── verdict: every cell is required ───────────────────────────────────────────
HARD_FAILED=""
while read -r c; do
  [ -z "$c" ] && continue
  HARD_FAILED="$HARD_FAILED $c"
done < "$FAILED"

if [ -n "$HARD_FAILED" ]; then
  log "[phase=failed] cells failed:$HARD_FAILED"
  exit 6
fi
log "[phase=done] all required cells complete -> $OUT"
exit 0
