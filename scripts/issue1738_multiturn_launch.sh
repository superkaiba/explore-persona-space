#!/usr/bin/env bash
# Issue #1738 multi-turn generate+capture — MULTI-POD / MULTI-GPU fan-out launcher
# (fork of scripts/issue779_ffc_n1m_launch.sh; parent pattern verbatim: one
# detached CVD-pinned process per GPU, pid breadcrumbs, per-shard logs; the
# workload RE-SHARDS off realized width via --shard-offset x --gpus-per-pod).
#
# Modes:
#   1. MANIFEST BUILD (once, CPU — cpu-mid lane, NOT a GPU pod):
#        bash scripts/issue1738_multiturn_launch.sh --build-manifest
#      Streams BOTH corpora to exhaustion (tiny-real probe first), allocates per
#      the pre-registered per-corpus rule, carves the pinned split, uploads the
#      manifest + split_1738.json to HF issue1738_multiturn/sampling_manifest/.
#   2. PILOT (G1 gate, ONE GPU of the fleet's realized type):
#        bash scripts/issue1738_multiturn_launch.sh --num-shards 32 --shard-offset 0 \
#          --gpus-per-pod 1 --pilot-cap 600
#      Pilot chunks count toward production (shard 0 resumes from them).
#   3. CAPTURE FAN-OUT (per instance i of 4, 8 GPUs each):
#        bash scripts/issue1738_multiturn_launch.sh --num-shards 32 --shard-offset $((8*i))
#   4. K-RESAMPLE (Phase 4a, one 8-GPU instance):
#        bash scripts/issue1738_multiturn_launch.sh --kresample --seeds 43,44,45,46 \
#          --subsample-file eval_results/issue_1738/kresample/kresample_subsample.json
#
# Pod-side: NO VM thread-cap prefix (dedicated GPUs keep full width).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Conditional .env source (GCE lane has no .env; the driver also load_dotenv()s).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

DRIVER="scripts/issue1738_multiturn_generate_capture.py"
LOG_DIR="${EPM_MT1738_LOG_DIR:-/workspace/logs}"
[ -d "$LOG_DIR" ] || LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

NUM_SHARDS=32
SHARD_OFFSET=0
GPUS_PER_POD=""
SHARD_SIZE="${EPM_MT1738_SHARD_SIZE:-500}"
PILOT_CAP=0
DRY_RUN=0
BUILD_MANIFEST=0
KRESAMPLE=0
SEEDS="43,44,45,46"
SUBSAMPLE_FILE=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --shard-offset) SHARD_OFFSET="$2"; shift 2 ;;
    --gpus-per-pod) GPUS_PER_POD="$2"; shift 2 ;;
    --shard-size) SHARD_SIZE="$2"; shift 2 ;;
    --pilot-cap) PILOT_CAP="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --build-manifest) BUILD_MANIFEST=1; shift ;;
    --kresample) KRESAMPLE=1; shift ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --subsample-file) SUBSAMPLE_FILE="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ── mode 1: build + upload the sampling manifest (foreground, CPU), then exit ─────
if [ "$BUILD_MANIFEST" -eq 1 ]; then
  cmd=(uv run python "$DRIVER" --build-sampling-manifest)
  [ ${#EXTRA_ARGS[@]} -gt 0 ] && cmd+=("${EXTRA_ARGS[@]}")
  echo "== issue1738 multi-turn manifest build (foreground, CPU) =="
  echo "  ${cmd[*]}"
  if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] no manifest built."; exit 0; fi
  "${cmd[@]}"
  echo "== manifest + split uploaded; run the capture fan-out with --manifest-from-hf =="
  exit 0
fi

# ── modes 2-4: GPU fan-out, G local shards CVD-pinned ─────────────────────────────
if [ -z "$GPUS_PER_POD" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS_PER_POD="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  [ -z "$GPUS_PER_POD" ] || [ "$GPUS_PER_POD" -lt 1 ] 2>/dev/null && GPUS_PER_POD=8
fi
if ! [ "$GPUS_PER_POD" -ge 1 ] 2>/dev/null; then
  echo "FATAL: could not resolve a positive --gpus-per-pod (got '$GPUS_PER_POD')" >&2
  exit 1
fi
if ! [ "$NUM_SHARDS" -ge 1 ] 2>/dev/null || ! [ "$SHARD_OFFSET" -ge 0 ] 2>/dev/null; then
  echo "FATAL: bad --num-shards ($NUM_SHARDS) / --shard-offset ($SHARD_OFFSET)" >&2
  exit 1
fi
LAST=$((SHARD_OFFSET + GPUS_PER_POD - 1))
if [ "$LAST" -ge "$NUM_SHARDS" ]; then
  echo "FATAL: shard-offset $SHARD_OFFSET + gpus-per-pod $GPUS_PER_POD exceeds --num-shards $NUM_SHARDS (last global index $LAST)" >&2
  exit 1
fi
if [ "$KRESAMPLE" -eq 1 ] && [ -z "$SUBSAMPLE_FILE" ]; then
  echo "FATAL: --kresample requires --subsample-file" >&2
  exit 1
fi

MODE="capture"
[ "$KRESAMPLE" -eq 1 ] && MODE="kresample"
echo "== issue1738 $MODE fan-out: pod owns global shards $SHARD_OFFSET..$LAST of $NUM_SHARDS (G=$GPUS_PER_POD, shard-size=$SHARD_SIZE, pilot-cap=$PILOT_CAP) =="

for g in $(seq 0 $((GPUS_PER_POD - 1))); do
  gidx=$((SHARD_OFFSET + g))
  log="$LOG_DIR/issue-1738-${MODE}-shard${gidx}.log"
  pidf="$LOG_DIR/issue-1738-${MODE}-shard${gidx}.pid"
  cmd=(uv run python "$DRIVER" --num-shards "$NUM_SHARDS" --shard-index "$gidx" --device cuda --shard-size "$SHARD_SIZE" --manifest-from-hf)
  if [ "$KRESAMPLE" -eq 1 ]; then
    cmd+=(--kresample --seeds "$SEEDS" --kresample-subsample "$SUBSAMPLE_FILE")
  fi
  if [ "$PILOT_CAP" -gt 0 ] 2>/dev/null; then
    cmd+=(--pilot-cap "$PILOT_CAP")
  fi
  [ ${#EXTRA_ARGS[@]} -gt 0 ] && cmd+=("${EXTRA_ARGS[@]}")
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  shard $gidx -> GPU $g | log=$log pid=$pidf"
    echo "    CUDA_VISIBLE_DEVICES=$g setsid nohup ${cmd[*]} > $log 2>&1 < /dev/null &"
    continue
  fi
  # $! after `setsid nohup ... &` is the intermediate; capture the real workload pid
  # via bash -c so the pidfile names the child, not the launcher subshell.
  PID=$(CUDA_VISIBLE_DEVICES=$g bash -c "setsid nohup ${cmd[*]} > $log 2>&1 < /dev/null & echo \$!")
  echo "$PID" > "$pidf"
  echo "[launch] shard $gidx -> GPU $g pid=$PID log=$log"
done

if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] no processes launched."; exit 0; fi
echo "== $GPUS_PER_POD shards launched on this pod; watch $LOG_DIR/issue-1738-${MODE}-shard*.log =="
