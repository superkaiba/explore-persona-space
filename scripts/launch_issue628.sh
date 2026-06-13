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
#   bash scripts/launch_issue628.sh [--phase {all,0b,1,2,3,4,finalize,resume}]
#                                   [--seeds 42,1042]
#
# Env knobs:
#   I628_SEEDS               seeds list (default 42,1042)
#   I628_MIN_TRAINED_CELLS   phase-1 coverage gate (default 30 of 56)
#   I628_MIN_PHASE2_CELLS    phase-2 coverage gate (default 1 G-cell)
#   EPM_I628_SKIP_PHASE_4    set to 1 to skip phase 4 entirely (#628 r13
#                            workaround for the open vLLM 0.11 + rsLoRA
#                            EngineCore crash). Phases 1+2+3 still
#                            finalize cleanly + emit epm:progress.
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

# Count Phase-2 G-eval cell JSONs landed on disk -- a cheap proxy for "some
# phase-2 work landed". Used by the post-phase-2 coverage gate so a phase-2
# crash that nonetheless produced cells does not kill phases 3/4.
count_phase2_cells() {
  find eval_results/issue_628/G_cells -type f -name '*__*__seed*.json' 2>/dev/null | wc -l
}

MIN_PHASE2_CELLS="${I628_MIN_PHASE2_CELLS:-1}"

# Coverage gate after phase 2: require at least MIN_PHASE2_CELLS G_cells to
# proceed. Below that the run is unanswerable; above it, phase 2 crashes are
# treated like phase 1 crashes -- record the gap, run phases 3/4 on what
# landed, and let the analyzer annotate missing cells (#628 r9).
gate_phase2_coverage() {
  local cells
  cells=$(count_phase2_cells)
  echo "[i628-launch] phase 2 coverage check: $cells G_cells landed (min=$MIN_PHASE2_CELLS)"
  if [ "$cells" -lt "$MIN_PHASE2_CELLS" ]; then
    echo "[i628-launch] phase 2 coverage below threshold; STOPPING (no Phase 3/4)"
    return 3
  fi
  return 0
}

# Phase-4 is the on-policy bystander reads (vLLM + LoRA). r9 hit a vLLM
# 0.11.0 + rsLoRA + punica-wrapper EngineCore crash during LoRA load
# (`peft_helper.py:55 Loading LoRA weights trained with rsLoRA.` followed by
# silent EngineCore_DP0 death and a downstream ZeroDivisionError in
# `vllm/entrypoints/llm.py:1610`'s pbar). This is upstream-of-our-code and
# enforce_eager=True is already on. Phase 4 is the SECONDARY measurement;
# the rig contrast (primary science) lives in phases 1+2+3 which are
# already done (4687 G-cell JSONs landed). r13 makes phase 4:
#   (a) skippable via EPM_I628_SKIP_PHASE_4=1 for the duration the vLLM
#       + rsLoRA crash is unresolved (escape hatch -- the run still
#       finalizes cleanly on phases 1+2+3 + epm:progress);
#   (b) wrapped in set +e so a crash does not kill _finalize -- without
#       the wrap the launcher exits rc=1 before _finalize fires and the
#       orchestrator never sees the partial-coverage sentinel.
SKIP_PHASE_4="${EPM_I628_SKIP_PHASE_4:-0}"

# Count phase-4 reads landed -- proxy for "phase 4 produced some output".
count_phase4_reads() {
  find eval_results/issue_628/bystander_onpolicy -type f -name 'reads.json' 2>/dev/null | wc -l
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

# Standalone _finalize: walks disk state, emits epm:results (full coverage)
# or epm:progress (partial) + [phase=done] iff full coverage. Used after a
# phase-4 crash so the orchestrator still sees the phase-1/2/3 results.
phase_finalize() {
  $PY scripts/i628_dispatch.py --phase finalize --seeds "$SEEDS"
}

echo "[i628-launch] starting (phase=$PHASE seeds=$SEEDS) at $(ts)"

run_one_phase() {
  case "$1" in
    0b) run_phase 0b phase_0b ;;
    1)  run_phase 1 phase_1 ;;
    2)  run_phase 2 phase_2 ;;
    3)  run_phase 3 phase_3 ;;
    4)  run_phase 4 phase_4 ;;
    finalize) run_phase finalize phase_finalize ;;
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

# Phase-4 driver shared by `all` and `resume`: phase 4 hits a vLLM 0.11 +
# rsLoRA EngineCore crash (#628 r9 + r13 post-mortem); SKIP_PHASE_4=1 lets
# the launcher skip it entirely so phases 1+2+3 still finalize cleanly. A
# non-skip phase-4 crash no longer kills _finalize -- the launcher always
# fires phase_finalize as the LAST step so the orchestrator sees the
# partial-coverage sentinel either way.
run_phase_4_with_tolerance() {
  if [ "$SKIP_PHASE_4" = "1" ]; then
    echo "[i628-launch] SKIP_PHASE_4=1 -- skipping phase 4 (vLLM+rsLoRA crash open in r13)"
    return 0
  fi
  set +e
  run_phase 4 phase_4
  local p4_rc=$?
  set -e
  local p4_reads
  p4_reads=$(count_phase4_reads)
  echo "[i628-launch] phase 4 reads landed: $p4_reads"
  if [ $p4_rc -ne 0 ]; then
    echo "[i628-launch] phase 4 rc=$p4_rc but continuing to _finalize (#628 r13)"
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
    # Phase 2: same tolerance shape as phase 1 (#628 r9). A worker SystemExit
    # (e.g. smoke-gate fail under --strict-gate, or some non-gate error) no
    # longer kills phases 3+4 as long as at least MIN_PHASE2_CELLS G_cells
    # landed. Phase 2 is idempotent (cell_p.exists() skip), so a rerun
    # processes only the missing cells.
    set +e
    run_phase 2 phase_2
    p2_rc=$?
    set -e
    gate_phase2_coverage || exit $?
    if [ $p2_rc -ne 0 ]; then
      echo "[i628-launch] phase 2 rc=$p2_rc but coverage gate PASSed; continuing"
    fi
    run_one_phase 3 || exit $?
    # Phase 4: tolerate crash; phase 4 is the secondary on-policy bystander
    # read, primary science is in phases 1+2+3 (#628 r13). The dispatcher's
    # _finalize fires at the tail of `--phase 4` regardless of phase 4's
    # own rc -- but only when the python process exits via main()'s normal
    # return path. A SystemExit raised from `_run_wave` skips that.
    run_phase_4_with_tolerance
    # Standalone finalize: writes the right sentinel even when phase 4
    # itself died before main()'s tail finalize. Idempotent vs the
    # dispatcher-emitted sentinel (last writer wins; coverage probes the
    # same disk state).
    run_one_phase finalize || exit $?
    ;;
  resume)
    # Resume: r5d/r8 post-mortem path. Phase 1 idempotently re-runs the
    # missing cells (completed cells skip via the per-cell sentinel); phase 2
    # re-runs with the same idempotency (cell_p.exists() skip), then 3/4.
    set +e
    run_phase 1 phase_1
    p1_rc=$?
    set -e
    gate_phase1_coverage || exit $?
    if [ $p1_rc -ne 0 ]; then
      echo "[i628-launch] phase 1 rc=$p1_rc but coverage gate PASSed; continuing"
    fi
    set +e
    run_phase 2 phase_2
    p2_rc=$?
    set -e
    gate_phase2_coverage || exit $?
    if [ $p2_rc -ne 0 ]; then
      echo "[i628-launch] phase 2 rc=$p2_rc but coverage gate PASSed; continuing"
    fi
    run_one_phase 3 || exit $?
    run_phase_4_with_tolerance
    run_one_phase finalize || exit $?
    ;;
  0b|1|2|3|4|finalize)
    run_one_phase "$PHASE"
    ;;
  *)
    echo "[i628-launch] unknown --phase $PHASE (use one of: all, resume, 0b, 1, 2, 3, 4)"
    exit 2
    ;;
esac

echo "[i628-launch] all requested phases done at $(ts)"
