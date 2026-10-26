#!/usr/bin/env bash
# Phase 0 workload sequencer for issue #2674 ("Does subliminal learning need
# multiple epochs? Repetition vs unique data vs steps").
#
# This is the `--workload-cmd` target of the plan's dispatch command:
#   uv run python scripts/dispatch_issue.py launch --issue 2674 \
#     --intent ft-7b --gpus 4 --repo-branch issue-2674 \
#     --boot-disk-gb 300 --time-budget-hours 24 \
#     --workload-cmd "bash scripts/issue2674_phase0_workload.sh"
#
# Order (each phase guarded by its own sentinel under $OUT_ROOT/sentinels, so a
# re-run after a crash keeps every completed phase's outputs):
#
#   preamble     GPU width + disk headroom, printed up front.
#   smoke        tiny end-to-end gate in a SEPARATE out-root: CPU unit checks,
#                every driver's --import-check, then a real 1-GPU pass through
#                generation -> reduce -> train -> sampled eval -> gate analysis.
#                Fails fast, before 300k prompts of generation are spent.
#   prompts      the ONE shared prompt list (CPU).
#   gen          teacher generation, sharded across the realized GPU count.
#   reduce       filter + dedup + shared-index intersection + held-out split (CPU).
#   upload_data  teacher pools + datagen metrics to the HF data repo.
#   train        the 9 Phase 0 training runs, as an N-wide GPU job queue.
#   eval_base    base-model sampled eval (shared by every pair; run once).
#   eval_runs    per-run sampled eval over that run's saved adapters.
#   gate         Phase 0 pass criteria -> metrics/gate_analysis.json + figures.
#   upload_final sampled completions + adapters + every metric JSON.
#
# The gate runs in --report-only mode so its verdict never blocks the uploads;
# this script re-reads gate_pass from the JSON at the end and exits 9 on a FAIL,
# AFTER everything is durable. Any other phase failing exits non-zero immediately
# (set -e), leaving its sentinel unwritten so a re-run resumes there.
#
# Env knobs:
#   OUT_ROOT              run out-root (default /workspace/issue2674_out)
#   EPS_2674_MAX_GPUS     clamp the job-queue width (default 4)
#   EPS_2674_N_PROMPTS    shared prompt-pool size (default 300000)
#   EPS_2674_UNIQUE       U for the Phase 0 runs (default 10000)
#   EPS_2674_SKIP_SMOKE   =1 skips the smoke stage (re-runs on a proven pod only)
#   FORCE_PHASES          space-separated phase names to re-run despite sentinels
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root, so scripts/ paths resolve

OUT_ROOT="${OUT_ROOT:-/workspace/issue2674_out}"
SMOKE_ROOT="${SMOKE_ROOT:-${OUT_ROOT}_smoke}"   # per-leg out-root: the smoke's
                                                # regime never contaminates the
                                                # production resume state.
LOG_DIR="$OUT_ROOT/logs"
SENT_DIR="$OUT_ROOT/sentinels"
N_PROMPTS="${EPS_2674_N_PROMPTS:-300000}"
UNIQUE="${EPS_2674_UNIQUE:-10000}"
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
  # An inherited pin already names PHYSICAL ids; re-exporting a literal 0..N-1
  # would retarget the workload onto unallocated devices (gotchas.md #1336).
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
  # Exclusive host (RunPod / GCE): device enumeration IS the allocation.
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

df -P "$OUT_ROOT"
$PY -c '
import sys
from pathlib import Path

from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

free = assert_out_root_headroom(Path(sys.argv[1]), 60, phase="issue2674-preamble")
print(f"[phase=preamble] out-root headroom OK: {free:.1f} GB free (need 60)")
' "$OUT_ROOT"

# ---------------------------------------------------------------------------
# N-wide GPU job queue: one process per GPU, CUDA_VISIBLE_DEVICES pinned in the
# LAUNCHER env of each child (an in-process pin is a driver-level no-op once any
# import has run cuInit). Every worker also passes --expect-single-gpu, so a pin
# that failed to take fails loud instead of co-locating.
#
# Args: <label> <cmd> [<cmd> ...]  — one shell command string per job.
# Returns 0 only when every job exited 0; prints every job's rc.
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
      [ -f "$qdir/job$j.rc" ] || continue
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

# ---------------------------------------------------------------------------
# smoke: CPU unit checks + every driver's --import-check + a real tiny 1-GPU
# pass through the WHOLE production chain, in its own out-root.
# ---------------------------------------------------------------------------
if [ "${EPS_2674_SKIP_SMOKE:-0}" = "1" ]; then
  echo "[phase=smoke] SKIPPED (EPS_2674_SKIP_SMOKE=1)"
elif phase_done smoke; then
  echo "[phase=smoke] already done — skip"
else
  echo "[phase=smoke] $(ts) SMOKE_ROOT=$SMOKE_ROOT"
  rm -rf "$SMOKE_ROOT"
  mkdir -p "$SMOKE_ROOT"

  # (1) CPU logic. --require-tokenizer because this lane has Hub access, so a
  #     skipped tokenizer check would be a silent coverage hole.
  $PY scripts/issue2674_unit_checks.py --require-tokenizer

  # (2) argparse-attribute completeness + shared-helper call-arity, per driver.
  for drv in gen_data train_run sampled_eval gate_analysis upload; do
    echo "[phase=smoke] import-check scripts/issue2674_$drv.py"
    $PY "scripts/issue2674_$drv.py" --import-check
  done

  SMOKE_GPU="${GPU_IDS[0]}"
  echo "[phase=smoke] tiny end-to-end on gpu $SMOKE_GPU"

  $PY scripts/issue2674_gen_data.py --phase prompts \
    --out "$SMOKE_ROOT" --n-prompts 96 --prompt-seed 42

  for teacher in cat neutral; do
    CUDA_VISIBLE_DEVICES="$SMOKE_GPU" $PY scripts/issue2674_gen_data.py \
      --phase generate --out "$SMOKE_ROOT" --teacher "$teacher" \
      --shard-index 0 --num-shards 1 --max-num-seqs 32 --expect-single-gpu
  done

  $PY scripts/issue2674_gen_data.py --phase reduce --out "$SMOKE_ROOT" \
    --num-shards 1 --heldout-size 4 --min-train-pool 8 --token-sample 32

  CUDA_VISIBLE_DEVICES="$SMOKE_GPU" $PY scripts/issue2674_train_run.py \
    --out "$SMOKE_ROOT" --teacher cat --unique 20 --epochs 1 \
    --schedule constant_with_warmup --seed 1 --run-name smoke_cat \
    --adapter-save-steps 0,1 --logp-every 1 --heldout-every 1 \
    --heldout-limit 4 --probe-batch-rows 44 --expect-single-gpu

  CUDA_VISIBLE_DEVICES="$SMOKE_GPU" $PY scripts/issue2674_sampled_eval.py \
    --out "$SMOKE_ROOT" --run-name smoke_cat --include-base \
    --question-limit 2 --sample-limit 2 --max-num-seqs 32 --expect-single-gpu

  # Gate analysis on a 1-run smoke can never PASS (8 of 9 runs absent); the smoke
  # asserts it RUNS and writes its JSON, not that it passes.
  $PY scripts/issue2674_gate_analysis.py --out "$SMOKE_ROOT" --unique 20 \
    --allow-missing-runs --report-only --no-figures

  echo "[phase=smoke] tiny end-to-end OK"
  mark_done smoke
fi

# ---------------------------------------------------------------------------
# prompts: the ONE shared prompt list, drawn without replacement.
# ---------------------------------------------------------------------------
if phase_done prompts; then
  echo "[phase=prompts] already done — skip"
else
  echo "[phase=prompts] $(ts) n=$N_PROMPTS"
  $PY scripts/issue2674_gen_data.py --phase prompts \
    --out "$OUT_ROOT" --n-prompts "$N_PROMPTS" --prompt-seed 42 \
    2>&1 | tee "$LOG_DIR/prompts.log"
  mark_done prompts
fi

# ---------------------------------------------------------------------------
# gen: both teachers, sharded across the realized GPU count.
#
# The shard count is FROZEN on first run (gen_shards.txt): resuming on a pod with
# a different GPU count must not re-shard, or the reduce phase's expected shard
# set stops matching what is on disk.
# ---------------------------------------------------------------------------
SHARDS_FILE="$SENT_DIR/gen_shards.txt"
if [ -f "$SHARDS_FILE" ]; then
  NUM_SHARDS="$(cat "$SHARDS_FILE")"
  echo "[phase=gen] reusing frozen shard count $NUM_SHARDS (realized gpu width $NGPU)"
else
  NUM_SHARDS="$NGPU"
  echo "$NUM_SHARDS" > "$SHARDS_FILE"
  echo "[phase=gen] shard count $NUM_SHARDS (= realized gpu width)"
fi

if phase_done gen; then
  echo "[phase=gen] already done — skip"
else
  echo "[phase=gen] $(ts) 2 teachers x $NUM_SHARDS shards"
  GEN_CMDS=()
  for teacher in cat neutral; do
    for ((s = 0; s < NUM_SHARDS; s++)); do
      # Per-shard .done.json makes each job idempotent: a resumed queue re-runs
      # only the shards that never completed.
      GEN_CMDS+=("uv run python scripts/issue2674_gen_data.py --phase generate \
--out '$OUT_ROOT' --teacher $teacher --shard-index $s --num-shards $NUM_SHARDS \
--expect-single-gpu")
    done
  done
  run_queue gen "${GEN_CMDS[@]}"
  mark_done gen
fi

# ---------------------------------------------------------------------------
# reduce: filter, dedup, intersect the teachers' surviving prompt indices, hold
# out the same 1k indices from both, measure tokens per training example.
# ---------------------------------------------------------------------------
if phase_done reduce; then
  echo "[phase=reduce] already done — skip"
else
  echo "[phase=reduce] $(ts)"
  $PY scripts/issue2674_gen_data.py --phase reduce --out "$OUT_ROOT" \
    --num-shards "$NUM_SHARDS" 2>&1 | tee "$LOG_DIR/reduce.log"
  mark_done reduce
fi

# ---------------------------------------------------------------------------
# upload_data: the teacher pools' rollout TEXT lands before any long training
# phase, so a crash in training never strands the generation spend.
# ---------------------------------------------------------------------------
if phase_done upload_data; then
  echo "[phase=upload_data] already done — skip"
else
  echo "[phase=upload_data] $(ts)"
  $PY scripts/issue2674_upload.py --out "$OUT_ROOT" --stage teacher_data \
    2>&1 | tee "$LOG_DIR/upload_teacher_data.log"
  mark_done upload_data
fi

# ---------------------------------------------------------------------------
# train: the 9 Phase 0 runs as an N-wide queue. The roster comes from the shared
# library (C.phase0_runs) so the shell and the gate analysis can never disagree
# about which runs exist.
# ---------------------------------------------------------------------------
mapfile -t RUN_SPECS < <($PY -c '
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue2674_common as C

for spec in C.phase0_runs(int(sys.argv[1])):
    print(f"{spec.name}\t{spec.teacher}\t{spec.epochs}\t{spec.schedule}\t{spec.seed}")
' "$UNIQUE")

if [ "${#RUN_SPECS[@]}" -ne 9 ]; then
  echo "FATAL: expected 9 Phase 0 runs, got ${#RUN_SPECS[@]}" >&2
  exit 2
fi
echo "[phase=train] roster: ${#RUN_SPECS[@]} runs, U=$UNIQUE"

if phase_done train; then
  echo "[phase=train] already done — skip"
else
  echo "[phase=train] $(ts)"
  TRAIN_CMDS=()
  for spec in "${RUN_SPECS[@]}"; do
    IFS=$'\t' read -r r_name r_teacher r_epochs r_sched r_seed <<< "$spec"
    if [ -f "$OUT_ROOT/runs/$r_name/run.done.json" ]; then
      echo "[phase=train] $r_name already complete — skip"
      continue
    fi
    TRAIN_CMDS+=("uv run python scripts/issue2674_train_run.py \
--out '$OUT_ROOT' --run-name $r_name --teacher $r_teacher --unique $UNIQUE \
--epochs $r_epochs --schedule $r_sched --seed $r_seed --expect-single-gpu")
  done
  run_queue train "${TRAIN_CMDS[@]}"
  mark_done train
fi

# ---------------------------------------------------------------------------
# eval_base: the base model's cat rate is shared by every pair — one engine, one
# pass, before the per-run queue so a base failure costs one GPU-minute.
# ---------------------------------------------------------------------------
if phase_done eval_base; then
  echo "[phase=eval_base] already done — skip"
else
  echo "[phase=eval_base] $(ts)"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" $PY scripts/issue2674_sampled_eval.py \
    --out "$OUT_ROOT" --base-only --expect-single-gpu \
    2>&1 | tee "$LOG_DIR/eval_base.log"
  mark_done eval_base
fi

# ---------------------------------------------------------------------------
# eval_runs: one vLLM process per run, serving ALL of that run's adapters through
# multi-LoRA in a single engine (one model load per run, not per checkpoint).
# ---------------------------------------------------------------------------
if phase_done eval_runs; then
  echo "[phase=eval_runs] already done — skip"
else
  echo "[phase=eval_runs] $(ts)"
  EVAL_CMDS=()
  for spec in "${RUN_SPECS[@]}"; do
    IFS=$'\t' read -r r_name _ _ _ _ <<< "$spec"
    if [ -f "$OUT_ROOT/runs/$r_name/eval.done.json" ]; then
      echo "[phase=eval_runs] $r_name already evaluated — skip"
      continue
    fi
    EVAL_CMDS+=("uv run python scripts/issue2674_sampled_eval.py \
--out '$OUT_ROOT' --run-name $r_name --max-loras 4 --expect-single-gpu")
  done
  run_queue eval_runs "${EVAL_CMDS[@]}"
  mark_done eval_runs
fi

# ---------------------------------------------------------------------------
# gate: every Phase 0 pass criterion + the figures. --report-only so a FAIL
# cannot abort the uploads below; the verdict is read back at the end.
# ---------------------------------------------------------------------------
if phase_done gate; then
  echo "[phase=gate] already done — skip"
else
  echo "[phase=gate] $(ts)"
  $PY scripts/issue2674_gate_analysis.py --out "$OUT_ROOT" --unique "$UNIQUE" \
    --report-only 2>&1 | tee "$LOG_DIR/gate.log"
  mark_done gate
fi

# ---------------------------------------------------------------------------
# upload_final: sampled completions, adapters, every metric JSON (gate included).
# ---------------------------------------------------------------------------
if phase_done upload_final; then
  echo "[phase=upload_final] already done — skip"
else
  echo "[phase=upload_final] $(ts)"
  $PY scripts/issue2674_upload.py --out "$OUT_ROOT" --stage sampled \
    2>&1 | tee "$LOG_DIR/upload_sampled.log"
  $PY scripts/issue2674_upload.py --out "$OUT_ROOT" --stage adapters \
    2>&1 | tee "$LOG_DIR/upload_adapters.log"
  $PY scripts/issue2674_upload.py --out "$OUT_ROOT" --stage metrics \
    2>&1 | tee "$LOG_DIR/upload_metrics.log"
  mark_done upload_final
fi

# ---------------------------------------------------------------------------
# Verdict. Everything is durable by now, so a gate FAIL exits non-zero without
# costing any artifact.
# ---------------------------------------------------------------------------
GATE_JSON="$OUT_ROOT/metrics/gate_analysis.json"
if [ ! -f "$GATE_JSON" ]; then
  echo "FATAL: $GATE_JSON missing after the gate phase" >&2
  exit 2
fi
GATE_RC=0
$PY -c '
import json
import sys

report = json.load(open(sys.argv[1]))
print("[phase=verdict] gate_pass =", report.get("gate_pass"))
for row in report.get("blockers", []):
    print("[phase=verdict] BLOCKER:", row)
for row in report.get("needs_user", []):
    print("[phase=verdict] NEEDS USER:", row)
sys.exit(0 if report.get("gate_pass") else 9)
' "$GATE_JSON" || GATE_RC=$?

echo "[phase=done] $(ts) gate_rc=$GATE_RC out_root=$OUT_ROOT"
exit "$GATE_RC"
