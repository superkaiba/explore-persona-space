#!/usr/bin/env bash
# Issue #2588-larger sbatch JOB BODY (fellows cluster "charmander").
#
# One job = ONE extension model, its arms run SEQUENTIALLY inside the job
# (arm a then arm b; glm53 is arm b only — arms come from the panel registry,
# never a second hand-kept list). Submit ONLY via
# scripts/issue2588x_submit.sh, which supplies the per-model resources
# (--gres/--cpus-per-task/--mem/--job-name/--output) on the sbatch command
# line and passes "<model_key> <tp_gpus>" as script arguments.
#
# Exact submit form (HF token fed via stdin so it never lands in shell
# history or on disk; it reaches the job through sbatch's DEFAULT environment
# export — no --export flag anywhere, and the token is never written to disk):
#
#   read -r HF_TOKEN; export HF_TOKEN; bash scripts/issue2588x_submit.sh q38fn
#
# Static, model-independent directives only — everything model-shaped rides
# the submit wrapper's sbatch flags:
#SBATCH -p general
#SBATCH --qos=high-eur
#SBATCH -t 36:00:00

set -euo pipefail

MODEL_KEY="${1:?usage: sbatch ... issue2588x_cell_job.sh <model_key> <tp_gpus>}"
TP="${2:?tp_gpus argument missing (issue2588x_submit.sh supplies it)}"
: "${HF_TOKEN:?HF_TOKEN must be in the submitting shell env}"

BASE=/workspace/superkaiba/eps2588x
export HF_HOME="$BASE/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_NVLS_ENABLE=0
export VLLM_GPU_MEM_UTIL=0.85
# Qwen3.8-Flash-Next runs at TP=4: at TP=2 the torch.compile autotune step OOMed (smoke jobs 61111 and
# 61212, 2026-09-03) beside 88 GiB of FP8 weights per GPU; VLLM_PLE_CPU_OFFLOAD is unknown to this vLLM nightly.
export PYTHONPATH="$BASE/repo/src:$BASE/repo/scripts"
PY="$BASE/venv/bin/python"
# vLLM JIT-compiles FlashInfer sampling kernels with `ninja`, which lives in the venv; it must be on PATH
# (same-width jobs 61187-61190 died at engine init with FileNotFoundError: ninja, 2026-09-03).
export PATH="$BASE/venv/bin:$PATH"
# Caching-allocator fragmentation was 21.8 GiB of the 136 GiB in use when the
# DeepSeek-V4-Flash capture OOMed (job 62667); expandable segments reclaim it.
# vLLM (gen phase) tolerates this setting outside sleep mode.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT_ROOT="$BASE/out"
mkdir -p "$OUT_ROOT" "$HF_HOME"
# NB: no `export CUDA_VISIBLE_DEVICES` anywhere in this job — Slurm's gres
# binding allocates the GPUs; vLLM tensor-parallel width rides --gpu-count.

# Single source of truth: the panel registry names the arms and pins the TP.
REG_TP="$("$PY" -c 'import sys, issue2588_panel_common as PC; print(PC.PANEL[sys.argv[1]].tp_gpus)' "$MODEL_KEY")"
if [ "$REG_TP" != "$TP" ]; then
  echo "[job] FATAL: tp argument $TP != registry tp_gpus $REG_TP for $MODEL_KEY" >&2
  exit 2
fi
# --- Node-local weight mirror (I/O only; results unaffected) ------------------------
# vLLM rebuilds its engine for every generation stage (G4/G5 regen passes, GPQA), and
# each rebuild re-reads the whole checkpoint. From MooseFS/FUSE that took ~19 min per
# build for the 173 GB Flash-Next FP8 snapshot (smoke 61323, 2026-09-03), so a cell pays
# hours in pure loading. Mirror the snapshot once to node-local /tmp (NVMe overlay) and
# point HF_HOME there when the node has room; otherwise fall back to the shared cache.
HF_ID="$("$PY" -c 'import sys, issue2588_panel_common as PC; print(PC.PANEL[sys.argv[1]].hf_id)' "$MODEL_KEY")"
REPO_DIR="models--${HF_ID//\//--}"
SRC="$BASE/hf_cache/hub/$REPO_DIR"
MIRROR_ROOT="/tmp/eps2588x_hf"
_eps_mirror_cleanup() { rm -rf "$MIRROR_ROOT/hub/$REPO_DIR" 2>/dev/null || true; }
if [ -d "$SRC" ]; then
  need_kb=$(du -sk "$SRC" | cut -f1)
  free_kb=$(df -Pk /tmp | awk 'NR==2{print $4}')
  if [ "$free_kb" -gt $(( need_kb * 12 / 10 )) ]; then
    echo "[job] $(date -u +%FT%TZ) mirroring $REPO_DIR ($(( need_kb / 1048576 )) GB) to $MIRROR_ROOT (free $(( free_kb / 1048576 )) GB)"
    mkdir -p "$MIRROR_ROOT/hub"
    if rsync -a "$SRC/" "$MIRROR_ROOT/hub/$REPO_DIR/"; then
      export HF_HOME="$MIRROR_ROOT"
      trap '_eps_mirror_cleanup' EXIT
      echo "[job] $(date -u +%FT%TZ) mirror done; HF_HOME=$HF_HOME"
    else
      echo "[job] $(date -u +%FT%TZ) mirror FAILED; falling back to shared HF_HOME" >&2
      _eps_mirror_cleanup
    fi
  else
    echo "[job] $(date -u +%FT%TZ) /tmp too small for a mirror (need $(( need_kb / 1048576 )) GB x1.2, free $(( free_kb / 1048576 )) GB); using shared HF_HOME"
  fi
fi

ARMS="$("$PY" -c 'import sys, issue2588_panel_common as PC; print(" ".join(PC.PANEL[sys.argv[1]].arms))' "$MODEL_KEY")"

term_handler() {
  echo "[trap] $(date -u +%FT%TZ) caught TERM/INT — killing the job's process group" >&2
  trap - TERM INT
  kill -- 0 2>/dev/null || true
  _eps_mirror_cleanup
  exit 143
}
trap term_handler TERM INT

(
  while true; do
    sleep 600
    echo "[hb] $(date -u +%FT%TZ) job=${SLURM_JOB_ID:-?} model=${MODEL_KEY} still running"
  done
) &
HB_PID=$!

for ARM in $ARMS; do
  echo "[job] $(date -u +%FT%TZ) launching cell ${MODEL_KEY}_${ARM} --phase all (tp=${TP})"
  "$PY" "$BASE/repo/scripts/issue2588_run_cell.py" \
    --cell "${MODEL_KEY}_${ARM}" \
    --phase all \
    --out-root "$OUT_ROOT" \
    --gpu-count "$TP" \
    --capture-batch-size "${EPS_CAPTURE_BS:-8}"
  echo "[job] $(date -u +%FT%TZ) cell ${MODEL_KEY}_${ARM} complete"
done

kill "$HB_PID" 2>/dev/null || true
echo "[job] $(date -u +%FT%TZ) all arms complete for ${MODEL_KEY} (${ARMS})"
