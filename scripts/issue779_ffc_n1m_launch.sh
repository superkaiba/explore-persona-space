#!/usr/bin/env bash
# Issue #779 n1m generate+capture — MULTI-POD / MULTI-GPU fan-out launcher (POD-side).
#
# The n1m corpus is captured across M pods x G GPUs = --num-shards total shards, all
# reading ONE HF-hosted sampling manifest. Each pod runs G local capture processes,
# each PINNED to its own GPU via CUDA_VISIBLE_DEVICES in the LAUNCHER environment
# (per-process env — NOT the Hydra +gpu_id path; see the CVD-clobber gotcha) and
# assigned a GLOBAL shard index = --shard-offset + local_gpu. Each child runs detached
# (setsid nohup < /dev/null) with its own log + pid breadcrumb, so the launcher returns
# immediately and the shards outlive the SSH session.
#
# Two modes:
#   1. MANIFEST BUILD (once, CPU — route to a cpu-bigmem/cpu-mid lane, NOT a GPU pod):
#        bash scripts/issue779_ffc_n1m_launch.sh --build-manifest
#      Streams LMSYS-to-exhaustion + WildChat top-up, runs the near-dupe gate, writes
#      + UPLOADS the manifest to HF issue779_monitoring/fitter-fair-comparison-n1m/
#      sampling_manifest/. Foreground; exits when done. Resumable (per-corpus cache).
#
#   2. CAPTURE FAN-OUT (per pod, GPU): assumes the manifest is on HF.
#        # pod 0 of 4 (8 GPUs each) -> global shards 0..7:
#        bash scripts/issue779_ffc_n1m_launch.sh --num-shards 32 --shard-offset 0
#        # pod 1 of 4 -> global shards 8..15:
#        bash scripts/issue779_ffc_n1m_launch.sh --num-shards 32 --shard-offset 8
#        # pod 2: --shard-offset 16 ; pod 3: --shard-offset 24
#        bash scripts/issue779_ffc_n1m_launch.sh --dry-run --shard-offset 8   # print, run nothing
#
# --gpus-per-pod defaults to the detected GPU count. Peak local footprint is bounded
# by the driver's per-chunk upload->verify->PURGE (~one in-flight chunk/process, <~1 GB;
# the trimmed capture is ~86 KB/context), so G parallel shards stay well under quota.
#
# Pod-side: NO VM thread-cap prefix (dedicated GPUs keep full width).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Conditional .env source (GCE lane has no .env; the driver also load_dotenv()s).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

DRIVER="scripts/issue779_ffc_n1m_generate_capture.py"
LOG_DIR="${EPM_N1M_LOG_DIR:-/workspace/logs}"
[ -d "$LOG_DIR" ] || LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

NUM_SHARDS=32
SHARD_OFFSET=0
GPUS_PER_POD=""
SHARD_SIZE="${EPM_N1M_SHARD_SIZE:-500}"
DRY_RUN=0
BUILD_MANIFEST=0
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --shard-offset) SHARD_OFFSET="$2"; shift 2 ;;
    --gpus-per-pod) GPUS_PER_POD="$2"; shift 2 ;;
    --shard-size) SHARD_SIZE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --build-manifest) BUILD_MANIFEST=1; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ── mode 1: build + upload the sampling manifest (foreground, CPU), then exit ─────
if [ "$BUILD_MANIFEST" -eq 1 ]; then
  cmd=(uv run python "$DRIVER" --build-sampling-manifest)
  [ ${#EXTRA_ARGS[@]} -gt 0 ] && cmd+=("${EXTRA_ARGS[@]}")
  echo "== issue779 n1m manifest build (foreground, CPU) =="
  echo "  ${cmd[*]}"
  if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] no manifest built."; exit 0; fi
  "${cmd[@]}"
  echo "== manifest built + uploaded to HF; run the capture fan-out per pod with --manifest-from-hf =="
  exit 0
fi

# ── mode 2: capture fan-out (GPU), G local shards CVD-pinned ──────────────────────
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
# Every global shard index this pod owns must be < NUM_SHARDS.
LAST=$((SHARD_OFFSET + GPUS_PER_POD - 1))
if [ "$LAST" -ge "$NUM_SHARDS" ]; then
  echo "FATAL: shard-offset $SHARD_OFFSET + gpus-per-pod $GPUS_PER_POD exceeds --num-shards $NUM_SHARDS (last global index $LAST)" >&2
  exit 1
fi

echo "== issue779 n1m capture fan-out: pod owns global shards $SHARD_OFFSET..$LAST of $NUM_SHARDS (G=$GPUS_PER_POD, shard-size=$SHARD_SIZE) =="

for g in $(seq 0 $((GPUS_PER_POD - 1))); do
  gidx=$((SHARD_OFFSET + g))
  log="$LOG_DIR/issue-779-n1m-shard${gidx}.log"
  pidf="$LOG_DIR/issue-779-n1m-shard${gidx}.pid"
  cmd=(uv run python "$DRIVER" --num-shards "$NUM_SHARDS" --shard-index "$gidx" --device cuda --shard-size "$SHARD_SIZE" --manifest-from-hf)
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
echo "== $GPUS_PER_POD shards launched on this pod; watch $LOG_DIR/issue-779-n1m-shard*.log =="
