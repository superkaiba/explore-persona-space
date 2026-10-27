#!/usr/bin/env bash
# Phase 1 (core) workload sequencer for issue #2674 ("Does subliminal learning need
# multiple epochs? Repetition vs unique data vs steps").
#
# Dispatch (auto lane, GCP first, RunPod fallback):
#   uv run python scripts/dispatch_issue.py launch --issue 2674 \
#     --intent ft-7b --gpus 4 --repo-branch issue-2674 \
#     --boot-disk-gb 300 --time-budget-hours 30 \
#     --workload-cmd "bash scripts/issue2674_core_workload.sh"
#
# The core: 3 conditions at 100k example-passes each (A: U=10k x 10 epochs, B: U=25k x 4,
# C: U=100k x 1), constant LR, seeds 1-5, cat + neutral = 30 runs, roster from
# C.core_runs(). It reuses the Phase 0 teacher pool and the Phase 0 base-model eval.
#
# Order (each phase guarded by a sentinel under $OUT_ROOT/sentinels):
#
#   preamble     GPU width + disk headroom.
#   stage_pool   Phase 0 pool from the HF data repo, sha-checked against its manifest.
#   smoke        CPU unit checks, every driver's --import-check, then a real tiny
#                1-GPU train -> sampled eval -> per-run upload (to the core_smoke
#                prefix) on the staged pool, in its own out-root.
#   runs         30 queue jobs over the GPU slots. Each job is one run's whole chain:
#                train -> sampled eval over its 13 adapters -> upload + verify. Each
#                step is skipped when its own done-file exists, so a re-run resumes.
#                Uploading per run means a pod death strands at most the in-flight runs.
#   upload_final logs, sentinels, every metric/eval JSON and completion, then an
#                exact-set verify of the data repo and of all adapters on the model repo.
#
# Env knobs:
#   OUT_ROOT              run out-root (default /workspace/issue2674_core, else $HOME/...)
#   EPS_2674_MAX_GPUS     clamp the job-queue width (default 4)
#   EPS_2674_SKIP_SMOKE   =1 skips the smoke stage (re-runs on a proven pod only)
#   FORCE_PHASES          space-separated phase names to re-run despite sentinels
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root, so scripts/ paths resolve

if [ -n "${OUT_ROOT:-}" ]; then
  :
elif [ -d /workspace ]; then
  OUT_ROOT=/workspace/issue2674_core
else
  OUT_ROOT="$HOME/issue2674_core"
fi
SMOKE_ROOT="${SMOKE_ROOT:-${OUT_ROOT}_smoke}"   # per-leg out-root
LOG_DIR="$OUT_ROOT/logs"
SENT_DIR="$OUT_ROOT/sentinels"
MAX_GPUS="${EPS_2674_MAX_GPUS:-4}"
FORCE_PHASES="${FORCE_PHASES:-}"
PY="uv run python"

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$SENT_DIR"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

phase_done() {
  local name="$1"
  for f in $FORCE_PHASES; do
    [ "$f" = "$name" ] && return 1
  done
  [ -f "$SENT_DIR/$name.done.json" ]
}

mark_done() {
  local name="$1"
  printf '{"phase": "%s", "completed_at": "%s"}\n' "$name" "$(ts)" \
    > "$SENT_DIR/$name.done.json"
  echo "[phase=$1] DONE $(ts)"
}

# ---------------------------------------------------------------------------
# preamble: GPU width (allocation-first), disk headroom
# ---------------------------------------------------------------------------
echo "[phase=preamble] $(ts) OUT_ROOT=$OUT_ROOT"

GPU_IDS=()
WIDTH_SOURCE=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
  WIDTH_SOURCE="CUDA_VISIBLE_DEVICES"
elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
  IFS=',' read -ra GPU_IDS <<< "$SLURM_JOB_GPUS"
  WIDTH_SOURCE="SLURM_JOB_GPUS"
elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
  IFS=',' read -ra GPU_IDS <<< "$SLURM_STEP_GPUS"
  WIDTH_SOURCE="SLURM_STEP_GPUS"
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  mapfile -t GPU_IDS < <(seq 0 $((SLURM_GPUS_ON_NODE - 1)))
  WIDTH_SOURCE="SLURM_GPUS_ON_NODE"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "FATAL: SLURM job $SLURM_JOB_ID exposes no GPU allocation env" >&2
  exit 2
else
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
  WIDTH_SOURCE="nvidia-smi"
fi

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "FATAL: no GPUs resolved (source=$WIDTH_SOURCE)" >&2
  exit 2
fi
if [ "${#GPU_IDS[@]}" -gt "$MAX_GPUS" ]; then
  GPU_IDS=("${GPU_IDS[@]:0:$MAX_GPUS}")
fi
NGPU="${#GPU_IDS[@]}"
echo "[phase=preamble] gpu width=$NGPU ids=${GPU_IDS[*]} source=$WIDTH_SOURCE"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# 30 runs x 13 adapters x ~80 MB = ~31 GB of adapters, plus completions and the pool.
df -P "$OUT_ROOT"
$PY -c '
import sys
from pathlib import Path

from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

free = assert_out_root_headroom(Path(sys.argv[1]), 60, phase="issue2674-core-preamble")
print(f"[phase=preamble] out-root headroom OK: {free:.1f} GB free (need 60)")
' "$OUT_ROOT"

# ---------------------------------------------------------------------------
# N-wide GPU job queue (same contract as the Phase 0 sequencer): one process per
# GPU, CUDA_VISIBLE_DEVICES pinned in the LAUNCHER env of each child; every worker
# passes --expect-single-gpu so a pin that failed to take fails loud.
#
# Args: <label> <cmd> [<cmd> ...]  — one shell command string per job.
# ---------------------------------------------------------------------------
run_queue() {
  local label="$1"; shift
  local -a cmds=("$@")
  local njobs="${#cmds[@]}"
  if [ "$njobs" -eq 0 ]; then
    echo "[queue:$label] nothing to run"
    return 0
  fi

  local qdir="$LOG_DIR/queue_$label"
  rm -rf "$qdir"
  mkdir -p "$qdir"

  local -a slot_job=()
  local -a slot_pid=()
  local i
  for ((i = 0; i < NGPU; i++)); do
    slot_job[i]=-1
    slot_pid[i]=0
  done

  local next=0 finished=0 failures=0 started=0
  echo "[queue:$label] $njobs job(s) over $NGPU gpu slot(s) $(ts)"

  while [ "$finished" -lt "$njobs" ]; do
    for ((i = 0; i < NGPU; i++)); do
      if [ "${slot_job[i]}" -eq -1 ] && [ "$next" -lt "$njobs" ]; then
        local j="$next"
        next=$((next + 1))
        started=$((started + 1))
        local gpu="${GPU_IDS[i]}"
        local logf="$LOG_DIR/${label}_job$(printf '%02d' "$j").log"
        local rcf="$qdir/job$j.rc"
        echo "[queue:$label] start job $((j + 1))/$njobs gpu=$gpu log=$logf $(ts)"
        echo "[queue:$label]   cmd: ${cmds[j]}"
        (
          set +e
          export CUDA_VISIBLE_DEVICES="$gpu"
          bash -c "${cmds[j]}" > "$logf" 2>&1
          echo "$?" > "$rcf"
        ) &
        slot_job[i]="$j"
        slot_pid[i]=$!
      fi
    done

    sleep 15

    for ((i = 0; i < NGPU; i++)); do
      local j="${slot_job[i]}"
      [ "$j" -eq -1 ] && continue
      if [ ! -f "$qdir/job$j.rc" ]; then
        # A wrapper killed before it wrote its rc (OOM killer) would spin the loop
        # forever; a vanished pid with no rc file is a failed job.
        if kill -0 "${slot_pid[i]}" 2>/dev/null; then
          continue
        fi
        sleep 2
        [ -f "$qdir/job$j.rc" ] || echo 255 > "$qdir/job$j.rc"
      fi
      local rc
      rc="$(cat "$qdir/job$j.rc")"
      wait "${slot_pid[i]}" 2>/dev/null || true
      finished=$((finished + 1))
      slot_job[i]=-1
      slot_pid[i]=0
      if [ "$rc" = "0" ]; then
        echo "[queue:$label] job $((j + 1))/$njobs OK ($finished/$njobs complete) $(ts)"
      else
        failures=$((failures + 1))
        echo "[queue:$label] job $((j + 1))/$njobs FAILED rc=$rc" \
          "log=$LOG_DIR/${label}_job$(printf '%02d' "$j").log $(ts)" >&2
        grep -iE 'error|traceback|killed|OOM|CUDA' \
          "$LOG_DIR/${label}_job$(printf '%02d' "$j").log" | tail -40 >&2 || true
      fi
    done
  done

  wait
  if [ "$failures" -ne 0 ]; then
    echo "[queue:$label] $failures/$njobs job(s) failed" >&2
    return 1
  fi
  echo "[queue:$label] all $njobs job(s) OK $(ts)"
  return 0
}

# Device-level free-memory gate on this job's pinned GPU, before each GPU step. Each
# slot alternates vLLM eval -> HF train ~8 times, so a leaked EngineCore (or a foreign
# holder that NVML shows no process row for) would OOM the next step; this waits out
# async teardown and otherwise fails the job loudly. Reads device memory.used, never
# per-process rows, because container pid namespaces hide the holder's pid.
gpu_free_gate() {
  local gpu="${CUDA_VISIBLE_DEVICES:?gpu_free_gate needs a pinned GPU}"
  local floor="${EPS_2674_GPU_FLOOR_MIB:-2048}" waited=0 used=""
  while :; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits \
      | tr -d ' ')" || used=""
    if [ -n "$used" ] && [ "$used" -le "$floor" ]; then
      echo "[gpu-gate] gpu=$gpu used=${used}MiB <= ${floor}MiB (waited ${waited}s)"
      return 0
    fi
    if [ "$waited" -ge 180 ]; then
      echo "[gpu-gate] FATAL gpu=$gpu holds ${used:-?}MiB after ${waited}s" >&2
      nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv >&2 || true
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
}
export -f gpu_free_gate

# One run's whole chain as a single queue-job command string. Each step is skipped
# when its own done-file exists; `set -e` stops the chain at the first failing step.
run_chain() {
  local root="$1" name="$2" teacher="$3" unique="$4" epochs="$5" sched="$6" seed="$7"
  local steps="$8" sub="$9" extra_train="${10:-}" extra_eval="${11:-}"
  printf '%s' "set -e; \
if [ -f '$root/runs/$name/run.done.json' ]; then echo '[chain] $name train already done'; else \
gpu_free_gate; \
uv run python scripts/issue2674_train_run.py --out '$root' --pool-dir '$OUT_ROOT/pool' \
--run-name $name --teacher $teacher --unique $unique --epochs $epochs --schedule $sched \
--seed $seed --adapter-save-steps $steps --expect-single-gpu $extra_train; fi; \
if [ -f '$root/sampled_eval/$name/eval.done.json' ]; then echo '[chain] $name eval already done'; else \
gpu_free_gate; \
uv run python scripts/issue2674_sampled_eval.py --out '$root' --run-name $name \
--max-loras 4 --expect-single-gpu $extra_eval; fi; \
if [ -f '$root/sentinels/upload_run_$name.done.json' ]; then echo '[chain] $name upload already done'; else \
uv run python scripts/issue2674_core_io.py upload-run --out '$root' --run-name $name \
--hf-subdir $sub; fi; \
echo '[chain] $name complete'"
}

# ---------------------------------------------------------------------------
# stage_pool: the Phase 0 teacher pool, byte-identical (sha-checked).
# ---------------------------------------------------------------------------
if phase_done stage_pool; then
  echo "[phase=stage_pool] already done — skip"
else
  echo "[phase=stage_pool] $(ts)"
  $PY scripts/issue2674_core_io.py stage-pool --out "$OUT_ROOT" \
    2>&1 | tee "$LOG_DIR/stage_pool.log"
  mark_done stage_pool
fi

# ---------------------------------------------------------------------------
# smoke: CPU logic + import checks + a real tiny chain through the new per-run
# upload path, in its own out-root and its own HF prefix (core_smoke).
# ---------------------------------------------------------------------------
if [ "${EPS_2674_SKIP_SMOKE:-0}" = "1" ]; then
  echo "[phase=smoke] SKIPPED (EPS_2674_SKIP_SMOKE=1)"
elif phase_done smoke; then
  echo "[phase=smoke] already done — skip"
else
  echo "[phase=smoke] $(ts) SMOKE_ROOT=$SMOKE_ROOT"
  if [ "$(realpath -m "$SMOKE_ROOT")" = "$(realpath -m "$OUT_ROOT")" ]; then
    echo "FATAL: SMOKE_ROOT equals OUT_ROOT ($OUT_ROOT); refusing to wipe it" >&2
    exit 2
  fi
  rm -rf "$SMOKE_ROOT"
  mkdir -p "$SMOKE_ROOT"

  $PY scripts/issue2674_unit_checks.py --require-tokenizer
  for drv in train_run sampled_eval core_io; do
    echo "[phase=smoke] import-check scripts/issue2674_$drv.py"
    $PY "scripts/issue2674_$drv.py" --import-check
  done

  SMOKE_GPU="${GPU_IDS[0]}"
  echo "[phase=smoke] tiny chain on gpu $SMOKE_GPU"
  SMOKE_CMD="$(run_chain "$SMOKE_ROOT" smoke_core_cat cat 20 2 constant_with_warmup 1 \
    1,2 core_smoke \
    "--logp-every 1 --heldout-every 1 --heldout-limit 4 --probe-batch-rows 44" \
    "--question-limit 2 --sample-limit 2 --max-num-seqs 32")"
  CUDA_VISIBLE_DEVICES="$SMOKE_GPU" bash -c "$SMOKE_CMD"
  echo "[phase=smoke] tiny chain OK"
  mark_done smoke
fi

# ---------------------------------------------------------------------------
# runs: the 30 core runs, roster from the shared library.
# ---------------------------------------------------------------------------
mapfile -t RUN_SPECS < <($PY -c '
import sys

sys.path.insert(0, "scripts")
import issue2674_common as C

for spec in C.core_runs():
    steps = ",".join(str(s) for s in spec.adapter_save_steps())
    print(f"{spec.name}\t{spec.teacher}\t{spec.unique}\t{spec.epochs}\t{spec.schedule}\t{spec.seed}\t{steps}")
')

if [ "${#RUN_SPECS[@]}" -ne 30 ]; then
  echo "FATAL: expected 30 core runs, got ${#RUN_SPECS[@]}" >&2
  exit 2
fi
echo "[phase=runs] roster: ${#RUN_SPECS[@]} runs"

if phase_done runs; then
  echo "[phase=runs] already done — skip"
else
  echo "[phase=runs] $(ts)"
  CHAIN_CMDS=()
  for spec in "${RUN_SPECS[@]}"; do
    IFS=$'\t' read -r r_name r_teacher r_unique r_epochs r_sched r_seed r_steps <<< "$spec"
    if [ -f "$SENT_DIR/upload_run_$r_name.done.json" ]; then
      echo "[phase=runs] $r_name already complete — skip"
      continue
    fi
    CHAIN_CMDS+=("$(run_chain "$OUT_ROOT" "$r_name" "$r_teacher" "$r_unique" "$r_epochs" \
      "$r_sched" "$r_seed" "$r_steps" core)")
  done
  run_queue runs "${CHAIN_CMDS[@]}"
  mark_done runs
fi

# ---------------------------------------------------------------------------
# upload_final: everything not already covered by a per-run upload.
# ---------------------------------------------------------------------------
if phase_done upload_final; then
  echo "[phase=upload_final] already done — skip"
else
  echo "[phase=upload_final] $(ts)"
  $PY scripts/issue2674_core_io.py upload-final --out "$OUT_ROOT" \
    2>&1 | tee "$LOG_DIR/upload_final.log"
  mark_done upload_final
fi

echo "[phase=done] $(ts) out_root=$OUT_ROOT"
exit 0
