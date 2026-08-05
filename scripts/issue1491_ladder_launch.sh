#!/usr/bin/env bash
# Task #1491: ladder generate+capture — MULTI-GPU fan-out launcher (POD-side / GCE-side).
#
# For ONE (scale, capture-mode) pair, runs the requested splits as SEQUENTIAL
# WAVES: each wave fans G shards across G local GPUs
# (CUDA_VISIBLE_DEVICES-pinned, per the CVD-clobber gotcha) with GLOBAL indices
# = --shard-offset + local_gpu, detached (setsid nohup < /dev/null), then
# BLOCKS until every shard pid of the wave is dead before launching the next
# split (gotchas.md "Chained waves on a detached-spawn launcher fan out
# CONCURRENTLY", #1738 — a detached-spawn launcher that exits after its spawn
# loop makes &&-chained waves simultaneous, N_waves × G engines per GPU). At
# most ONE driver per GPU is live at any time.
#
# BECAUSE the launcher now blocks for the whole sweep, launch IT detached too
# (the standard pod-side shape): setsid nohup bash scripts/issue1491_ladder_launch.sh
# ... > /workspace/logs/issue-1491-launch-<scale>.log 2>&1 < /dev/null &
# The orchestrator polls the pod-side sentinel / per-phase log breadcrumbs;
# markers post from the VM side.
#
# Ladder-specific vs the parent (#779) launcher:
#   * --scale {0.5B, 1.5B, 3B, 7B, 14B, 32B}: resolves --model + --layers +
#     --hf-prefix; plan §4.2 depth-fraction-mapped layers per scale.
#   * --split <name> | --all-splits: one of {train_25k, val_400, test_1000,
#     wc_test_1k, tierB_3600, ceiling_draw_43, ceiling_draw_44}, or the ordered
#     sweep default (train_25k, then val/test, then wc_test, then ceiling draws).
#   * --capture-mode coresident (default; ≤7B) | phase_split_gen | phase_split_capture.
#     14B/32B run TWO passes: --capture-mode phase_split_gen (vLLM gen only,
#     raw completions to HF), THEN --capture-mode phase_split_capture (HF
#     model only — no vLLM engine — one teacher-forced forward per persisted
#     response, joined by context id; resume keys on the .pt alone).
#     `launch.sh ... phase_split_gen && launch.sh ... phase_split_capture`
#     is VALID sequencing: this launcher blocks (wait_wave_dead) until every
#     shard pid of every wave is dead before exiting, so the capture
#     invocation starts only after the last gen shard has died and its GPU
#     memory is released. RC CONTRACT (round-3a MINOR / M3 enabler): exit 0
#     ONLY when every shard of every wave reached [phase=done]; a PARTIAL
#     wave (some shards failed) exits 3 and a fully-failed wave exits 1 —
#     so the `&&` chain refuses to run capture over a broken gen wave
#     (the driver's Hub-required join is the second, structural guard).
#   * ENV KNOBS exported explicitly (plan §11 + parent driver commit
#     4cb9d6ea8d): EPM_VLLM_ENFORCE_EAGER=1, EPM_VLLM_DISABLE_PREFIX_CACHING=1.
#     Never assume defaults — the ENV-gated knobs are OFF unless exported.
#   * HF Hub accelerators: HF_HUB_ENABLE_HF_TRANSFER=1 + HF_XET_HIGH_PERFORMANCE=1
#     are re-asserted here so the GCE lane (which has no bootstrap_pod.sh) also
#     inherits the fast upload path (upload-policy §HF-uploads-accelerated-by-default).
#
# Example — pod 0 of 1 (8 GPUs) doing all splits for the 0.5B rung:
#   setsid nohup bash scripts/issue1491_ladder_launch.sh --scale 0.5B --all-splits \
#        --num-shards 8 --shard-offset 0 \
#        > /workspace/logs/issue-1491-launch-scale05.log 2>&1 < /dev/null &
#
# Example — 32B rung across 2× 8-GPU pods (16 shards total), gen phase:
#   pod A: (same detached shape) --scale 32B --all-splits \
#            --capture-mode phase_split_gen --num-shards 16 --shard-offset 0
#   pod B: (same, --shard-offset 8)
#   then per pod, after the gen waves drain (chainable in one detached
#   command via `&&` — the launcher blocks until its waves are dead):
#            --capture-mode phase_split_capture with IDENTICAL
#            --num-shards/--shard-offset/--shard-size — the capture wave's
#            ci join asserts the gen wave's shard arithmetic verbatim.
#
# Pod-side: NO VM thread-cap prefix (dedicated GPUs keep full width).
#
# GPU width + the per-shard CUDA_VISIBLE_DEVICES values come from the SLURM
# allocation env on the fellows lane (SLURM_JOB_ID set) and from nvidia-smi
# enumeration elsewhere — see "GPU discovery" below (#1902 shared-node gotcha).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"

# Conditional .env source (GCE lane has no .env; the driver also load_dotenv()s).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

# ---- ENV knobs (plan §11 + parent driver commit 4cb9d6ea8d) ----------------
# H100 long-prompt hang / IMA mitigation, ENV-gated in the parent driver.
export EPM_VLLM_ENFORCE_EAGER="${EPM_VLLM_ENFORCE_EAGER:-1}"
export EPM_VLLM_DISABLE_PREFIX_CACHING="${EPM_VLLM_DISABLE_PREFIX_CACHING:-1}"

# ---- HF Hub upload accelerators (upload-policy default-ON) -----------------
# Re-assert in this shell — the GCE lane has no bootstrap_pod.sh to set them
# and the huggingface_hub constants freeze HF_HUB_ENABLE_HF_TRANSFER at
# import time. RunPod pods already get these from bootstrap_pod.sh; setting
# them again is idempotent.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# WandB project pin for --workload-cmd launches (plan-consistency; irrelevant
# in this driver — no training — but keeps run-visibility uniform in case a
# nested tool logs).
export WANDB_PROJECT="${WANDB_PROJECT:-issue1491}"

DRIVER="scripts/issue1491_ladder_generate_capture.py"
LOG_DIR="${EPM_LADDER_LOG_DIR:-/workspace/logs}"
[ -d "$LOG_DIR" ] || LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# ---- CLI parsing -----------------------------------------------------------

SCALE=""
SPLIT=""
ALL_SPLITS=0
CAPTURE_MODE="coresident"
CAPTURE_BATCH_SIZE=8
NUM_SHARDS=8
SHARD_OFFSET=0
GPUS_PER_POD=""
SHARD_SIZE="${EPM_LADDER_SHARD_SIZE:-500}"
FIRST_CHUNK_SELF_GATE=0
DRY_RUN=0
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --scale) SCALE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --all-splits) ALL_SPLITS=1; shift ;;
    --capture-mode) CAPTURE_MODE="$2"; shift 2 ;;
    --capture-batch-size) CAPTURE_BATCH_SIZE="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --shard-offset) SHARD_OFFSET="$2"; shift 2 ;;
    --gpus-per-pod) GPUS_PER_POD="$2"; shift 2 ;;
    --shard-size) SHARD_SIZE="$2"; shift 2 ;;
    --first-chunk-self-gate) FIRST_CHUNK_SELF_GATE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [ -z "$SCALE" ]; then
  echo "FATAL: --scale is required (one of 0.5B 1.5B 3B 7B 14B 32B)" >&2
  exit 2
fi

# ---- Per-scale resolution (plan §4.2) -------------------------------------

MODEL=""
LAYERS=""
SCALE_SLUG=""
case "$SCALE" in
  0.5B) MODEL="Qwen/Qwen2.5-0.5B-Instruct"; LAYERS="12,16,22"; SCALE_SLUG="scale05" ;;
  1.5B) MODEL="Qwen/Qwen2.5-1.5B-Instruct"; LAYERS="14,19,26"; SCALE_SLUG="scale15" ;;
  3B)   MODEL="Qwen/Qwen2.5-3B-Instruct";   LAYERS="18,24,33"; SCALE_SLUG="scale3"  ;;
  7B)   MODEL="Qwen/Qwen2.5-7B-Instruct";   LAYERS="14,19,26"; SCALE_SLUG="scale7_refit" ;;
  14B)  MODEL="Qwen/Qwen2.5-14B-Instruct";  LAYERS="24,33,45"; SCALE_SLUG="scale14" ;;
  32B)  MODEL="Qwen/Qwen2.5-32B-Instruct";  LAYERS="32,43,59"; SCALE_SLUG="scale32" ;;
  *)
    echo "FATAL: unknown --scale '$SCALE' (expected one of 0.5B 1.5B 3B 7B 14B 32B)" >&2
    exit 2
    ;;
esac
HF_PREFIX="issue1491_scale_ladder/$SCALE_SLUG"

# ---- Split resolution ------------------------------------------------------

# Default sweep order (per plan §4.2): the generation-side splits, then the
# ceiling draws (seed 43, seed 44 on the SAME 1,000 test contexts).
DEFAULT_SPLITS=(train_25k val_400 test_1000 wc_test_1k tierB_3600 ceiling_draw_43 ceiling_draw_44)

SPLITS_TO_RUN=()
if [ "$ALL_SPLITS" -eq 1 ]; then
  SPLITS_TO_RUN=("${DEFAULT_SPLITS[@]}")
elif [ -n "$SPLIT" ]; then
  SPLITS_TO_RUN=("$SPLIT")
else
  echo "FATAL: pass either --split <name> or --all-splits" >&2
  exit 2
fi

# ---- GPU discovery + shard arithmetic --------------------------------------
#
# Width + PHYSICAL device ids. On a SLURM job (SLURM_JOB_ID set — the fellows
# H200 lane) the node is GPU-SHARED with other tenants and `nvidia-smi -L`
# ALWAYS enumerates all 8 physical devices (it ignores CUDA_VISIBLE_DEVICES),
# so a detected-count fan-out over-shards onto other tenants' GPUs (#1902
# crash 1; gotchas.md "Fellows SLURM nodes are GPU-SHARED"). On SLURM, width +
# ids therefore come from the allocation env via the landed #1902 reference
# implementation scripts/issue1902_common.py::realized_gpu_ids (REUSED, not
# reimplemented — same shell-out shape as scripts/issue1902_dispatch.sh):
# CUDA_VISIBLE_DEVICES (slurm-set) > SLURM_JOB_GPUS / SLURM_STEP_GPUS >
# SLURM_GPUS_ON_NODE (count only; ids assumed 0..N-1), clamped to
# SLURM_GPUS_ON_NODE (`-clamped` source suffix), FAIL LOUD when a SLURM job
# exposes none of the three — never the physical nvidia-smi count. Non-SLURM
# lanes (RunPod / GCE exclusive hosts) keep the nvidia-smi enumeration +
# fallback-8 unchanged (ids 0..N-1, exactly the prior behavior). An explicit
# --gpus-per-pod always wins on WIDTH; on SLURM its ids are still drawn from
# the allocation (first K, `-override` source suffix) and it FAILS LOUD when
# it exceeds the allocation — never 0..K-1 blindly.

GPU_SOURCE=""
GPU_IDS=()
if [ -n "${SLURM_JOB_ID:-}" ]; then
  GPU_LINE="$(uv run python -c '
import os, sys
sys.path.insert(0, "scripts")
import issue1902_common as C
src, ids = C.realized_gpu_ids(os.environ, 0)
assert src != "detected", "launcher bug: SLURM branch entered without SLURM_JOB_ID"
print(src + "|" + ",".join(ids))
')" || {
    echo "FATAL: SLURM GPU derivation failed (SLURM_JOB_ID=${SLURM_JOB_ID} set but realized_gpu_ids refused — traceback above); refusing the nvidia-smi physical count on a shared node (#1902)" >&2
    exit 2
  }
  GPU_SOURCE="${GPU_LINE%%|*}"
  IFS=',' read -r -a GPU_IDS <<< "${GPU_LINE##*|}"
  if [ "${#GPU_IDS[@]}" -lt 1 ]; then
    echo "FATAL: empty GPU id list from the SLURM allocation derivation (line: '$GPU_LINE')" >&2
    exit 2
  fi
  if [ -n "$GPUS_PER_POD" ]; then
    if ! [ "$GPUS_PER_POD" -ge 1 ] 2>/dev/null || [ "$GPUS_PER_POD" -gt "${#GPU_IDS[@]}" ]; then
      echo "FATAL: --gpus-per-pod $GPUS_PER_POD invalid or exceeds the SLURM allocation (${#GPU_IDS[@]} GPUs: ${GPU_IDS[*]}) — refusing to fan out onto other tenants' devices" >&2
      exit 2
    fi
    GPU_IDS=("${GPU_IDS[@]:0:GPUS_PER_POD}")
    GPU_SOURCE="${GPU_SOURCE}-override"
  fi
  GPUS_PER_POD="${#GPU_IDS[@]}"
else
  if [ -z "$GPUS_PER_POD" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      GPUS_PER_POD="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    fi
    # if [ empty or 0 ] fall back to 8 (the default 8-GPU pod shape)
    if ! [ "${GPUS_PER_POD:-0}" -ge 1 ] 2>/dev/null; then GPUS_PER_POD=8; fi
    GPU_SOURCE="detected"
  else
    GPU_SOURCE="override"
  fi
  for i in $(seq 0 $((GPUS_PER_POD - 1))); do GPU_IDS+=("$i"); done
fi
# The derivation-source token below is the fix-engaged signal for the #1902
# shared-node class: on the fellows lane it must read slurm-*, never detected.
echo "[gpu-derivation] source=$GPU_SOURCE gpus_per_pod=$GPUS_PER_POD ids=$(IFS=','; echo "${GPU_IDS[*]}")"
if ! [ "$NUM_SHARDS" -ge 1 ] 2>/dev/null || ! [ "$SHARD_OFFSET" -ge 0 ] 2>/dev/null; then
  echo "FATAL: bad --num-shards ($NUM_SHARDS) / --shard-offset ($SHARD_OFFSET)" >&2
  exit 2
fi
LAST=$((SHARD_OFFSET + GPUS_PER_POD - 1))
if [ "$LAST" -ge "$NUM_SHARDS" ]; then
  echo "FATAL: shard-offset $SHARD_OFFSET + gpus-per-pod $GPUS_PER_POD exceeds --num-shards $NUM_SHARDS (last global index $LAST)" >&2
  exit 2
fi

echo "== issue1491 ladder launch =="
echo "  scale=$SCALE  model=$MODEL  layers=$LAYERS  hf_prefix=$HF_PREFIX"
echo "  splits=${SPLITS_TO_RUN[*]}  capture_mode=$CAPTURE_MODE  batch=$CAPTURE_BATCH_SIZE"
echo "  num_shards=$NUM_SHARDS  shard_offset=$SHARD_OFFSET  gpus_per_pod=$GPUS_PER_POD  shard_size=$SHARD_SIZE"
echo "  ENV: EPM_VLLM_ENFORCE_EAGER=$EPM_VLLM_ENFORCE_EAGER EPM_VLLM_DISABLE_PREFIX_CACHING=$EPM_VLLM_DISABLE_PREFIX_CACHING"
echo "       HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER HF_XET_HIGH_PERFORMANCE=$HF_XET_HIGH_PERFORMANCE"
echo "  log_dir=$LOG_DIR"

# Extra flag(s) passed through to the driver.
DRIVER_EXTRAS=()
if [ "$FIRST_CHUNK_SELF_GATE" -eq 1 ]; then
  DRIVER_EXTRAS+=(--first-chunk-self-gate)
fi
[ ${#EXTRA_ARGS[@]} -gt 0 ] && DRIVER_EXTRAS+=("${EXTRA_ARGS[@]}")

# ---- Wave helpers -----------------------------------------------------------

# A shard pid is LIVE iff /proc/<pid> exists AND its cmdline still names the
# driver (guards PID reuse across multi-hour waves; a `uv run` wrapper's
# cmdline carries the driver path, and if uv execs into python the cmdline
# still does).
_shard_live() {
  local p="$1"
  [ -n "$p" ] || return 1
  [ -d "/proc/$p" ] || return 1
  tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "issue1491_ladder_generate_capture" || return 1
  return 0
}

# Block until every pid passed is dead. This poll loop is what makes the
# per-split waves SEQUENTIAL: the shards are setsid-detached (reparented to
# pid 1), so bash `wait` cannot apply and a spawn-and-exit launcher would fan
# every wave out concurrently (gotchas.md "Chained waves on a detached-spawn
# launcher", #1738 — 16 shards over 8 GPUs within seconds, MooseFS wedge).
wait_wave_dead() {
  local poll=30
  local hb=20   # heartbeat every hb polls (~10 min) — no silent multi-hour phases
  local n=0
  local total=$#
  while :; do
    local live=0
    local p
    for p in "$@"; do
      if _shard_live "$p"; then live=$((live + 1)); fi
    done
    [ "$live" -eq 0 ] && break
    if [ $((n % hb)) -eq 0 ]; then
      echo "[wave] $(date -u +%Y-%m-%dT%H:%M:%SZ) live=$live/$total shards; polling every ${poll}s"
    fi
    n=$((n + 1))
    sleep "$poll"
  done
}

# ---- Per-split SEQUENTIAL waves, per-shard fan-out within each wave ---------

ABORT=0
PARTIAL=0
for split_name in "${SPLITS_TO_RUN[@]}"; do
  if [ "$ABORT" -eq 1 ]; then break; fi
  echo "-- split=$split_name --"
  WAVE_PIDS=()
  WAVE_LOGS=()
  WAVE_IDX=()
  for g in $(seq 0 $((GPUS_PER_POD - 1))); do
    gidx=$((SHARD_OFFSET + g))
    # CVD pin = the PHYSICAL device id from the derivation above (== g on
    # non-SLURM lanes); shard arithmetic stays on the LOCAL index g.
    dev="${GPU_IDS[$g]}"
    log="$LOG_DIR/issue-1491-${SCALE_SLUG}-${split_name}-shard${gidx}.log"
    pidf="$LOG_DIR/issue-1491-${SCALE_SLUG}-${split_name}-shard${gidx}.pid"
    cmd=(
      uv run python "$DRIVER"
        --model "$MODEL"
        --layers "$LAYERS"
        --split "$split_name"
        --hf-prefix "$HF_PREFIX"
        --capture-mode "$CAPTURE_MODE"
        --capture-batch-size "$CAPTURE_BATCH_SIZE"
        --num-shards "$NUM_SHARDS"
        --shard-index "$gidx"
        --shard-size "$SHARD_SIZE"
        --device cuda
        --verbose
    )
    [ ${#DRIVER_EXTRAS[@]} -gt 0 ] && cmd+=("${DRIVER_EXTRAS[@]}")

    # %q-quote the argv so `bash -c` re-parses it verbatim (round-3a MINOR:
    # ${cmd[*]} word-splits on IFS — safe today only because every arg is
    # space-free; %q makes the re-parse structurally safe).
    cmd_q=$(printf '%q ' "${cmd[@]}")

    if [ "$DRY_RUN" -eq 1 ]; then
      echo "  shard $gidx -> GPU $dev | log=$log pid=$pidf"
      echo "    CUDA_VISIBLE_DEVICES=$dev setsid nohup $cmd_q> $log 2>&1 < /dev/null &"
      continue
    fi

    # $! after `setsid nohup ... &` is the intermediate; capture the real
    # workload pid via bash -c so the pidfile names the child, not the
    # launcher subshell (parent parity). The log path rides as a positional
    # param (not interpolated) so a space in $LOG_DIR cannot split it.
    PID=$(CUDA_VISIBLE_DEVICES=$dev bash -c "setsid nohup $cmd_q> \"\$1\" 2>&1 < /dev/null & echo \$!" _ "$log")
    echo "$PID" > "$pidf"
    echo "[launch] scale=$SCALE_SLUG split=$split_name shard=$gidx -> GPU $dev pid=$PID log=$log"
    WAVE_PIDS+=("$PID")
    WAVE_LOGS+=("$log")
    WAVE_IDX+=("$gidx")
  done

  if [ "$DRY_RUN" -eq 1 ]; then continue; fi

  wait_wave_dead "${WAVE_PIDS[@]}"

  # Wave verdict: count terminal [phase=done] breadcrumbs (C.phase("done")).
  n_done=0
  failed=()
  for i in "${!WAVE_LOGS[@]}"; do
    if grep -q "\[phase=done\]" "${WAVE_LOGS[$i]}" 2>/dev/null; then
      n_done=$((n_done + 1))
    else
      failed+=("${WAVE_IDX[$i]}")
    fi
  done
  if [ "$n_done" -eq 0 ]; then
    echo "FATAL: split=$split_name — 0/${#WAVE_LOGS[@]} shards reached [phase=done]; systemic failure, aborting remaining splits" >&2
    ABORT=1
  elif [ "${#failed[@]}" -gt 0 ]; then
    PARTIAL=1
    echo "WARNING: split=$split_name — shards missing [phase=done]: ${failed[*]} (${n_done}/${#WAVE_LOGS[@]} done); continuing to next split (shards are independent; resume re-runs the gaps)" >&2
  else
    echo "[wave] split=$split_name complete: ${n_done}/${#WAVE_LOGS[@]} shards done"
  fi
done

if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] no processes launched."; exit 0; fi
if [ "$ABORT" -eq 1 ]; then
  echo "== aborted after a fully-failed wave; logs at $LOG_DIR/issue-1491-${SCALE_SLUG}-*.log ==" >&2
  exit 1
fi
if [ "$PARTIAL" -eq 1 ]; then
  # Round-3a MINOR (M3 enabler): a distinct non-zero rc so a chained
  # `... phase_split_gen && ... phase_split_capture` cannot proceed over a
  # gen wave with failed shards. Resume by re-running the SAME command —
  # the Hub-resume predicate skips completed chunks.
  echo "== sweep finished with PARTIAL failures (shards missing [phase=done] above); rc=3 — a chained capture wave must not consume an incomplete gen wave; re-run this command to resume the gaps. Logs at $LOG_DIR/issue-1491-${SCALE_SLUG}-*.log ==" >&2
  exit 3
fi
echo "== ordered sweep complete; logs at $LOG_DIR/issue-1491-${SCALE_SLUG}-*.log =="
