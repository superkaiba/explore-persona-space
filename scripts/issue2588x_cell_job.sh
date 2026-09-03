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
# Qwen3.8-Flash-Next (qwen4_exp): keep the 51 GB n-gram embedding table in host RAM.
# Smoke job 61111 (2026-09-03) OOMed at torch.compile autotune with it on-GPU at TP=2;
# read only by the qwen4_exp model code, inert for the other families.
export VLLM_PLE_CPU_OFFLOAD=1
export PYTHONPATH="$BASE/repo/src:$BASE/repo/scripts"
PY="$BASE/venv/bin/python"
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
ARMS="$("$PY" -c 'import sys, issue2588_panel_common as PC; print(" ".join(PC.PANEL[sys.argv[1]].arms))' "$MODEL_KEY")"

term_handler() {
  echo "[trap] $(date -u +%FT%TZ) caught TERM/INT — killing the job's process group" >&2
  trap - TERM INT
  kill -- 0 2>/dev/null || true
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
    --gpu-count "$TP"
  echo "[job] $(date -u +%FT%TZ) cell ${MODEL_KEY}_${ARM} complete"
done

kill "$HB_PID" 2>/dev/null || true
echo "[job] $(date -u +%FT%TZ) all arms complete for ${MODEL_KEY} (${ARMS})"
