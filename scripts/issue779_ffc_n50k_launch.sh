#!/usr/bin/env bash
# Issue #779 n50k generate+capture — multi-GPU fan-out launcher (POD-side).
#
# Builds the sampling manifest ONCE (foreground, GPU 0), then fans out K capture
# processes, each PINNED to its own GPU via CUDA_VISIBLE_DEVICES in the LAUNCHER
# environment (per-process env — NOT the Hydra +gpu_id path; see the CVD-clobber
# gotcha). Each child runs detached (setsid nohup < /dev/null) with its own log
# file + pid breadcrumb, so the launcher returns immediately and the shards
# outlive the SSH session. K=8 by default (derived from nvidia-smi if unset);
# works at K=4.
#
# Peak local footprint is bounded by the driver's per-chunk upload->verify->PURGE
# (each process holds ~one in-flight chunk, < ~4 GB), so K parallel shards stay
# well under the ~130 GB MooseFS per-pod quota / the < 60 GB target.
#
# Target pod: pod-77950 (8xH100), the fresh provision per the 2026-07-14 scope
# update (pod-779 is not resumed). This launcher hardcodes NO SSH/host details —
# it runs ON the pod; the name is for the operator's logs/breadcrumbs only.
#
# Usage (pod, repo root):
#   bash scripts/issue779_ffc_n50k_launch.sh                 # K = detected GPU count
#   bash scripts/issue779_ffc_n50k_launch.sh --num-shards 4  # explicit K=4
#   bash scripts/issue779_ffc_n50k_launch.sh --dry-run       # print cmds+env, run nothing
#
# Pod-side: NO VM thread-cap prefix (dedicated GPUs keep full width).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Conditional .env source (GCE lane has no .env; the driver also load_dotenv()s).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

DRIVER="scripts/issue779_ffc_n50k_generate_capture.py"
LOG_DIR="${EPM_N50K_LOG_DIR:-/workspace/logs}"
[ -d "$LOG_DIR" ] || LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

NUM_SHARDS=""
DRY_RUN=0
SHARD_SIZE="${EPM_N50K_SHARD_SIZE:-500}"
BATCH_SIZE="${EPM_N50K_BATCH_SIZE:-16}"
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --shard-size) SHARD_SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# Derive K from the visible GPU count when not given.
if [ -z "$NUM_SHARDS" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_SHARDS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  [ -z "$NUM_SHARDS" ] || [ "$NUM_SHARDS" -lt 1 ] 2>/dev/null && NUM_SHARDS=8
fi
if ! [ "$NUM_SHARDS" -ge 1 ] 2>/dev/null; then
  echo "FATAL: could not resolve a positive --num-shards (got '$NUM_SHARDS')" >&2
  exit 1
fi

MANIFEST_CMD=(uv run python "$DRIVER" --build-sampling-manifest --shard-size "$SHARD_SIZE" --batch-size "$BATCH_SIZE")
[ ${#EXTRA_ARGS[@]} -gt 0 ] && MANIFEST_CMD+=("${EXTRA_ARGS[@]}")

echo "== issue779 n50k fan-out: K=$NUM_SHARDS shards, shard-size=$SHARD_SIZE, log-dir=$LOG_DIR =="

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] manifest build (foreground, GPU 0):"
  echo "  CUDA_VISIBLE_DEVICES=0 ${MANIFEST_CMD[*]}"
  echo "[dry-run] then $NUM_SHARDS detached capture shards:"
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    log="$LOG_DIR/issue-779-n50k-shard${i}.log"
    pidf="$LOG_DIR/issue-779-n50k-shard${i}.pid"
    cmd=(uv run python "$DRIVER" --num-shards "$NUM_SHARDS" --shard-index "$i" --device cuda --shard-size "$SHARD_SIZE" --batch-size "$BATCH_SIZE")
    [ ${#EXTRA_ARGS[@]} -gt 0 ] && cmd+=("${EXTRA_ARGS[@]}")
    echo "  shard $i -> GPU $i | log=$log pid=$pidf"
    echo "    CUDA_VISIBLE_DEVICES=$i setsid nohup ${cmd[*]} > $log 2>&1 < /dev/null &"
  done
  echo "[dry-run] no processes launched."
  exit 0
fi

# 1) Build the sampling manifest ONCE (foreground) — every shard reads it.
echo "[manifest] building sampling_manifest.json (foreground) ..."
CUDA_VISIBLE_DEVICES=0 "${MANIFEST_CMD[@]}"

# 2) Fan out K detached capture shards, each CVD-pinned to its own GPU.
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  log="$LOG_DIR/issue-779-n50k-shard${i}.log"
  pidf="$LOG_DIR/issue-779-n50k-shard${i}.pid"
  cmd=(uv run python "$DRIVER" --num-shards "$NUM_SHARDS" --shard-index "$i" --device cuda --shard-size "$SHARD_SIZE" --batch-size "$BATCH_SIZE")
  [ ${#EXTRA_ARGS[@]} -gt 0 ] && cmd+=("${EXTRA_ARGS[@]}")
  # $! after `setsid nohup ... &` is the intermediate; capture the real workload
  # pid via bash -c so the pidfile names the child, not the launcher subshell.
  PID=$(CUDA_VISIBLE_DEVICES=$i bash -c "setsid nohup ${cmd[*]} > $log 2>&1 < /dev/null & echo \$!")
  echo "$PID" > "$pidf"
  echo "[launch] shard $i -> GPU $i pid=$PID log=$log"
done

echo "== all $NUM_SHARDS shards launched; watch $LOG_DIR/issue-779-n50k-shard*.log =="
