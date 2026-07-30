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
#   5. BARE-QUERY CAPTURE (follow-up `bare-query` B1, plan §4.1 — forward-only,
#      uploads to issue1738_multiturn/bare_query, never the parent prefix):
#        bash scripts/issue1738_multiturn_launch.sh --bare-query --num-shards 8 --shard-offset 0
#   6. SAE-ARM CAPTURE (follow-up `sae-arm` S1, plan v8 §4.3 — teacher-forced
#      SAE forwards over the PARENT chunks; uploads to issue1738_multiturn/sae_arm,
#      never the parent prefix). Runs the G-S0/G-S1 pilot FOREGROUND on GPU 0
#      first (a designed-halt rc 26/27 aborts the launcher BEFORE the fleet
#      detaches — the fitness kill-gate contract), then fans out all shards
#      (shard 0 resumes past its pilot chunks via the Hub index):
#        bash scripts/issue1738_multiturn_launch.sh --sae-arm --num-shards 8 --shard-offset 0
#   7. CROSSED MANIFEST (fu3 P0, plan v9 §4.1 — foreground CPU; reads the MAIN
#      manifest from HF, writes issue1738_crossed/sampling_manifest):
#        bash scripts/issue1738_multiturn_launch.sh --crossed-manifest --manifest-from-hf
#   8. CROSSED CAPTURE (fu3 S1, plan v9 §4.2 — shard BY PREFIX; uploads to
#      issue1738_crossed/*, never the parent prefix). Runs the G1/G2/SAE pilot
#      FOREGROUND on GPU 0 first (designed-halt rc 28/29/30 aborts the launcher
#      BEFORE the fleet detaches), then fans out all shards (shard 0 resumes
#      past its pilot chunks via the Hub index):
#        bash scripts/issue1738_multiturn_launch.sh --crossed --num-shards 8 --shard-offset 0
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
BARE_QUERY=0
SAE_ARM=0
CROSSED=0
CROSSED_MANIFEST=0
CROSSED_PILOT_PREFIXES="${EPM_MT1738_CROSSED_PILOT_PREFIXES:-100}"
BARE_UPLOAD_PREFIX="issue1738_multiturn/bare_query"
SAE_UPLOAD_PREFIX="issue1738_multiturn/sae_arm"
SAE_PILOT_ROWS="${EPM_MT1738_SAE_PILOT_ROWS:-2000}"
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
    --bare-query) BARE_QUERY=1; shift ;;
    --bare-upload-prefix) BARE_UPLOAD_PREFIX="$2"; shift 2 ;;
    --sae-arm) SAE_ARM=1; shift ;;
    --crossed) CROSSED=1; shift ;;
    --crossed-manifest) CROSSED_MANIFEST=1; shift ;;
    --crossed-pilot-prefixes) CROSSED_PILOT_PREFIXES="$2"; shift 2 ;;
    --sae-upload-prefix) SAE_UPLOAD_PREFIX="$2"; shift 2 ;;
    --sae-pilot-rows) SAE_PILOT_ROWS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --subsample-file) SUBSAMPLE_FILE="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [ $((BARE_QUERY + KRESAMPLE + SAE_ARM + CROSSED + CROSSED_MANIFEST)) -gt 1 ]; then
  echo "FATAL: --bare-query / --kresample / --sae-arm / --crossed / --crossed-manifest are mutually exclusive" >&2
  exit 1
fi
if [ "$SAE_ARM" -eq 1 ]; then
  DRIVER="scripts/issue1738_sae_arm.py"
fi

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

# ── mode 7: build + upload the CROSSED manifest (fu3 P0; foreground, CPU) ─────────
if [ "$CROSSED_MANIFEST" -eq 1 ]; then
  cmd=(uv run python "$DRIVER" --build-crossed-manifest)
  [ ${#EXTRA_ARGS[@]} -gt 0 ] && cmd+=("${EXTRA_ARGS[@]}")
  echo "== issue1738 CROSSED manifest build (fu3 P0; foreground, CPU) =="
  echo "  ${cmd[*]}"
  if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] no crossed manifest built."; exit 0; fi
  "${cmd[@]}"
  echo "== crossed manifest uploaded; run --crossed with --manifest-from-hf (implied) =="
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
[ "$BARE_QUERY" -eq 1 ] && MODE="bare"
[ "$SAE_ARM" -eq 1 ] && MODE="sae"
[ "$CROSSED" -eq 1 ] && MODE="crossed"
echo "== issue1738 $MODE fan-out: pod owns global shards $SHARD_OFFSET..$LAST of $NUM_SHARDS (G=$GPUS_PER_POD, shard-size=$SHARD_SIZE, pilot-cap=$PILOT_CAP) =="

# sae-arm G-S0/G-S1 pilot: FOREGROUND on GPU 0 BEFORE the fleet detaches — a
# designed-halt rc (26 rate fence / 27 fitness kill) propagates through set -e
# and aborts the launcher, so the fleet never proceeds past a G-S0 FAIL
# (plan v8 §7; idempotent: a PASS meta already on the Hub skips the pilot).
if [ "$SAE_ARM" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
  echo "== sae-arm pilot (G-S0/G-S1) foreground on GPU 0, shard $SHARD_OFFSET =="
  CUDA_VISIBLE_DEVICES=0 uv run python "$DRIVER" --phase capture \
    --num-shards "$NUM_SHARDS" --shard-index "$SHARD_OFFSET" --device cuda \
    --sae-hf-prefix "$SAE_UPLOAD_PREFIX" --pilot-rows "$SAE_PILOT_ROWS" --pilot-only \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  echo "== sae-arm pilot PASS — fanning out $GPUS_PER_POD shards =="
fi

# crossed G1/G2/SAE pilot: FOREGROUND on GPU 0 BEFORE the fleet detaches — a
# designed-halt rc (28 G1 fence / 29 violation rate / 30 G2 sanity) propagates
# through set -e and aborts the launcher, so the fleet never proceeds past a
# pilot FAIL (plan v9 §7; the pilot ALSO publishes crossed_pilot_meta.json to
# the Hub — the fleet shards' authoritative sae_enabled verdict).
if [ "$CROSSED" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
  echo "== crossed pilot (G1/G2/SAE) foreground on GPU 0, shard $SHARD_OFFSET =="
  CUDA_VISIBLE_DEVICES=0 uv run python "$DRIVER" --crossed-capture \
    --num-shards "$NUM_SHARDS" --shard-index "$SHARD_OFFSET" --device cuda \
    --shard-size "$SHARD_SIZE" --manifest-from-hf \
    --pilot-cap "$CROSSED_PILOT_PREFIXES" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  echo "== crossed pilot PASS — fanning out $GPUS_PER_POD shards =="
fi

for g in $(seq 0 $((GPUS_PER_POD - 1))); do
  gidx=$((SHARD_OFFSET + g))
  log="$LOG_DIR/issue-1738-${MODE}-shard${gidx}.log"
  pidf="$LOG_DIR/issue-1738-${MODE}-shard${gidx}.pid"
  if [ "$SAE_ARM" -eq 1 ]; then
    # plan v8 §4.3: sae uploads ride their own prefix; the parent capture
    # prefix is never written by this mode (read-side only).
    cmd=(uv run python "$DRIVER" --phase capture --num-shards "$NUM_SHARDS" --shard-index "$gidx" --device cuda --sae-hf-prefix "$SAE_UPLOAD_PREFIX")
  elif [ "$CROSSED" -eq 1 ]; then
    # plan v9 §4.2: crossed uploads ride issue1738_crossed (the driver default);
    # the parent capture prefix is never written by this mode.
    cmd=(uv run python "$DRIVER" --crossed-capture --num-shards "$NUM_SHARDS" --shard-index "$gidx" --device cuda --shard-size "$SHARD_SIZE" --manifest-from-hf)
  else
    cmd=(uv run python "$DRIVER" --num-shards "$NUM_SHARDS" --shard-index "$gidx" --device cuda --shard-size "$SHARD_SIZE" --manifest-from-hf)
  fi
  if [ "$KRESAMPLE" -eq 1 ]; then
    cmd+=(--kresample --seeds "$SEEDS" --kresample-subsample "$SUBSAMPLE_FILE")
  fi
  if [ "$BARE_QUERY" -eq 1 ]; then
    # plan §4.1.4: bare uploads ride their own prefix; the parent capture
    # prefix is never written by this mode.
    cmd+=(--bare-query --upload-prefix "$BARE_UPLOAD_PREFIX")
  fi
  if [ "$PILOT_CAP" -gt 0 ] 2>/dev/null && [ "$SAE_ARM" -eq 0 ] && [ "$CROSSED" -eq 0 ]; then
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
