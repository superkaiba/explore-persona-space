#!/usr/bin/env bash
# Issue #628 marker-rig revision launcher.
#
# The original `/workspace/i628-launch-r5d.sh` chained
#   phase 0b -> 1 -> 2 -> 3 -> 4
# under `set -e`. When the Phase-1 band-stop callback OOM'd on the long-
# context cells (`icl_k8`, `reph_polite`, ...) two of four workers died with
# `worker failures: [(0, 1), (1, 1)]`, set -e propagated to the chain, and
# Phases 2/3/4 never ran -- only 36 of 56 adapters trained and the rest of
# the experiment stalled (#628 r5d post-mortem; addressed in r6 by
# (a) chunking the band-stop probe forward in `MarkerBandStopCallback`
# and (b) running Phases 2/3/4 with `--partial-ok` so a partial Phase-1
# completion drives downstream phases on the cells that *did* train).
#
# This launcher captures Phase 1's exit code, requires at least
# `I628_MIN_TRAINED_CELLS` trained cells (default 30 of 56 = ~54%), and
# fires Phases 2/3/4 with `--partial-ok` on what landed. Re-running this
# launcher after a Phase-1 OOM is safe: Phase-1's per-cell (adapter,
# stop_step) sentinel makes the train pass idempotent, so completed cells
# skip and only the missing 20 (or fewer, after the band-stop chunk fix)
# enter the training queue.
#
# Usage:
#   bash scripts/launch_issue628.sh [--phase {all,0b,1,2,3,4,resume}]
#                                   [--seeds 42,1042]
#
# The default `--phase all` runs the full chain; `--phase resume` is the
# documented relaunch shape for r5d -- it re-runs Phase 1 (skipping the 36
# completed cells, training only the 20 missing), then Phases 2/3/4 with
# `--partial-ok`.
#
set -uo pipefail   # NOT `-e`: we want explicit per-phase rc handling.

cd /workspace/explore-persona-space
export REPO_ROOT=/workspace/explore-persona-space
export WORKLOAD_ROOT=/workspace/explore-persona-space
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
# vLLM 0.11.0 V1 EngineCore dies silently 1-4s after init under default fork()
# if the parent main() touched CUDA-adjacent code before LLM(); spawn fixes it.
# (.claude/rules/gotchas.md "vLLM 0.11.0 V1 EngineCore fork() silent-death",
# committed via 48835909c.)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p /workspace/logs

PY=/workspace/explore-persona-space/.venv/bin/python
SEEDS="${I628_SEEDS:-42,1042}"
MIN_TRAINED_CELLS="${I628_MIN_TRAINED_CELLS:-30}"
PHASE="${1:-all}"
case "$PHASE" in
  --phase)
    PHASE="${2:?missing phase}"
    ;;
esac

ts() { date -u +%FT%TZ; }

count_trained_cells() {
  ls eval_results/issue_628/p1/stop_steps/*.json 2>/dev/null | wc -l
}

run_phase() {
  local label="$1"
  shift
  echo "[i628-launch] phase $label starting at $(ts)"
  "$@"
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[i628-launch] phase $label complete at $(ts)"
  else
    echo "[i628-launch] phase $label FAILED rc=$rc at $(ts)"
  fi
  return $rc
}

# Phase 0b: vLLM-driven base-response prefetch (one process, vLLM-only).
phase_0b() {
  CUDA_VISIBLE_DEVICES=0 $PY scripts/i628_dispatch.py \
    --phase 0b --seeds "$SEEDS" --enforce-gate
}

# Phase 1: 4-way wave-train (one worker per GPU). May leave a partial result
# if some cells OOM; the launcher's gate below decides whether to continue.
phase_1() {
  $PY scripts/i628_dispatch.py --phase 1 --seeds "$SEEDS" --enforce-gate
}

# Phases 2/3/4: --partial-ok skips cells whose adapter never trained instead
# of crashing on hf_hub_download. Phase 3 reuses external #537 adapters and
# is unaffected by partial Phase 1.
phase_2() {
  $PY scripts/i628_dispatch.py --phase 2 --seeds "$SEEDS" --enforce-gate --partial-ok
}
phase_3() {
  $PY scripts/i628_dispatch.py --phase 3 --seeds "$SEEDS" --enforce-gate
}
phase_4() {
  CUDA_VISIBLE_DEVICES=0 $PY scripts/i628_dispatch.py \
    --phase 4 --seeds "$SEEDS" --enforce-gate --partial-ok
}

echo "[i628-launch] starting (phase=$PHASE seeds=$SEEDS) at $(ts)"

run_one_phase() {
  case "$1" in
    0b) run_phase 0b phase_0b ;;
    1)  run_phase 1 phase_1 ;;
    2)  run_phase 2 phase_2 ;;
    3)  run_phase 3 phase_3 ;;
    4)  run_phase 4 phase_4 ;;
    *)  echo "[i628-launch] unknown phase $1"; return 2 ;;
  esac
}

# Coverage gate: after Phase 1 we require >= MIN_TRAINED_CELLS adapters to
# proceed; below that the run is structurally unanswerable and the chain
# aborts. Above it, --partial-ok lets Phases 2/3/4 process what trained and
# the clean-result analysis annotates the missing cells explicitly.
gate_phase1_coverage() {
  local trained
  trained=$(count_trained_cells)
  echo "[i628-launch] phase 1 coverage check: $trained / 56 cells trained (min=$MIN_TRAINED_CELLS)"
  if [ "$trained" -lt "$MIN_TRAINED_CELLS" ]; then
    echo "[i628-launch] coverage below threshold; STOPPING (no Phase 2/3/4)"
    return 3
  fi
  return 0
}

case "$PHASE" in
  all)
    run_one_phase 0b || exit $?
    # Phase 1: tolerate non-zero rc as long as coverage >= MIN_TRAINED_CELLS.
    # This is the crux of the r6 launcher: a worker OOM no longer kills the
    # downstream phases.
    set +e
    run_phase 1 phase_1
    p1_rc=$?
    set -e
    gate_phase1_coverage || exit $?
    if [ $p1_rc -ne 0 ]; then
      echo "[i628-launch] phase 1 rc=$p1_rc but coverage gate PASSed; continuing"
    fi
    run_one_phase 2 || exit $?
    run_one_phase 3 || exit $?
    run_one_phase 4 || exit $?
    ;;
  resume)
    # Resume: r5d post-mortem path. Phase 1 idempotently re-runs the missing
    # cells (completed cells skip via the per-cell sentinel), then 2/3/4.
    set +e
    run_phase 1 phase_1
    p1_rc=$?
    set -e
    gate_phase1_coverage || exit $?
    if [ $p1_rc -ne 0 ]; then
      echo "[i628-launch] phase 1 rc=$p1_rc but coverage gate PASSed; continuing"
    fi
    run_one_phase 2 || exit $?
    run_one_phase 3 || exit $?
    run_one_phase 4 || exit $?
    ;;
  0b|1|2|3|4)
    run_one_phase "$PHASE"
    ;;
  *)
    echo "[i628-launch] unknown --phase $PHASE (use one of: all, resume, 0b, 1, 2, 3, 4)"
    exit 2
    ;;
esac

echo "[i628-launch] all requested phases done at $(ts)"
