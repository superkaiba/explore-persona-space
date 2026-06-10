#!/usr/bin/env bash
# Issue #541 sweep dispatcher — prior-stratified rerun of #500.
#
# Smoke IS the sweep with one cell (PASS_UNIFIED): `--smoke` runs THIS SAME
# script end to end with ARMS=(marine_biologist), SEEDS=(42), a 4-candidate
# prescreen, and a 2-bystander eval slice — every phase flows through the
# identical `uv run python scripts/<entrypoint>.py` subprocess shape, env
# injection, logging surface ([phase=...] lines), and teardown. The full
# sweep is the same dispatcher uncapped.
#
# poll_pipeline.py contract: one `[phase=<name>]` line per logical phase and
# a terminal `[phase=done]` AFTER the results sentinel is written (the
# wrapper's `--phase results-sentinel` writes
# /workspace/logs/issue-541-epm_results-<epoch>.json with the
# _SENTINEL_REQUIRED_KEYS). Pod-side code NEVER shells out to scripts/task.py.
#
# Re-entrancy: every phase is idempotent (skip-if-output-exists / per-row
# judge resume), so a crashed run is resumed by re-launching this script.
#
# Usage (pod, from the repo root):
#   nohup bash scripts/run_issue541_sweep.sh > /workspace/logs/issue-541-sweep.log 2>&1 &
#   nohup bash scripts/run_issue541_sweep.sh --smoke > /workspace/logs/issue-541-smoke.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  export EPM_541_SMOKE=1
fi
# EPM_541_DRYRUN=1: structural dry-run — every python invocation is echoed
# instead of executed, so the dispatcher's phase sequencing, wave scheduler,
# env injection, and terminal [phase=done] are exercisable on a GPU-less VM
# (the carve-out's dispatcher dry-run). Implies the smoke arm set.
DRYRUN="${EPM_541_DRYRUN:-0}"
if [[ "$DRYRUN" == "1" ]]; then
  SMOKE=1
  export EPM_541_SMOKE=1
fi

# MooseFS quota mitigations (upload-policy rule; adapters publish inline via
# TrainLoraConfig.hf_upload, merged dirs are deleted after each cell's eval).
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

N_GPUS="${EPM_541_N_GPUS:-4}"
# Smoke artifacts are namespaced under issue_541_smoke so they never poison
# the full run's skip-if-exists resume logic on the same pod.
EVAL_ROOT_NAME="issue_541"
if [[ "$SMOKE" == "1" ]]; then EVAL_ROOT_NAME="issue_541_smoke"; fi
SCREEN_JSON="eval_results/$EVAL_ROOT_NAME/phase0_prescreen/prior_screen.json"
SMOKE_FLAG=()
if [[ "$SMOKE" == "1" ]]; then SMOKE_FLAG=(--smoke); fi

log_phase() { echo "[phase=$1] $(date -u +%FT%TZ)"; }
export DRYRUN

runpy() {
  # Single execution chokepoint for every python invocation: real run by
  # default, echo-only under EPM_541_DRYRUN=1 (same command line either way).
  if [[ "$DRYRUN" == "1" ]]; then
    echo "[dryrun] $*"
  else
    "$@"
  fi
}

run_wave() {
  # Run the queued commands (one per line on stdin) <= N_GPUS at a time, each
  # with CUDA device = its slot id substituted for the literal {GPU}.
  #
  # GPU pinning is belt-and-braces (incident 2026-06-10, bug_class
  # gpu_pinning_cvd_late: every wave job landed on physical GPU 0 → OOM):
  #   1. CUDA_VISIBLE_DEVICES=$gpu is exported in the spawned job's env, so
  #      whenever CUDA initializes in the child — even before sft.py's
  #      per-process clobber runs — the visible device IS the assigned one.
  #   2. --gpu-id {GPU} stays on every job line; sft.py:~960 re-writes
  #      CUDA_VISIBLE_DEVICES=str(gpu_id), the IDENTICAL string, so the two
  #      mechanisms agree no matter when CUDA initializes.
  local fail=0
  local -a pids=()
  local slot=0
  while IFS= read -r cmd; do
    [[ -z "$cmd" ]] && continue
    local gpu=$((slot % N_GPUS))
    local final_cmd="${cmd//\{GPU\}/$gpu}"
    echo "[wave] gpu=$gpu :: CUDA_VISIBLE_DEVICES=$gpu $final_cmd"
    CUDA_VISIBLE_DEVICES=$gpu bash -c "$(declare -f runpy); $final_cmd" &
    pids+=("$!")
    slot=$((slot + 1))
    if (( ${#pids[@]} >= N_GPUS )); then
      for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
      pids=()
    fi
  done
  for pid in "${pids[@]:-}"; do [[ -n "$pid" ]] && { wait "$pid" || fail=1; }; done
  if (( fail )); then
    echo "[wave] FAILED — at least one job exited non-zero" >&2
    return 1
  fi
}

# ── preflight ────────────────────────────────────────────────────────────────
log_phase preflight
# Non-fatal: a branch-pinned pod is perpetually "behind origin/main" (task
# markers land on main every few minutes), which preflight counts as an
# ERROR → exit 1 → set -e killed the dispatcher silently (incident
# 2026-06-10, two dead smoke launches). The summary still prints loudly;
# the wrapper preflight phase below remains the fatal experiment gate.
runpy uv run python -m explore_persona_space.orchestrate.preflight \
  || echo "[preflight] non-fatal failure (see summary above; branch pods are perpetually behind origin/main)"
runpy uv run python scripts/run_experiment_541.py --arm marine_biologist --phase preflight --gpu-id 0 "${SMOKE_FLAG[@]}"

# ── Phase 0: prescreen (one process per step — vLLM/HF never share a proc) ──
log_phase prescreen
runpy uv run python scripts/issue541_prescreen.py --step 0a --gpu-id 0 "${SMOKE_FLAG[@]}"
runpy uv run python scripts/issue541_prescreen.py --step 0b-gen --gpu-id 0 "${SMOKE_FLAG[@]}"
runpy uv run python scripts/issue541_prescreen.py --step 0b-judge "${SMOKE_FLAG[@]}"
runpy uv run python scripts/issue541_prescreen.py --step 0c --gpu-id 0 "${SMOKE_FLAG[@]}"
runpy uv run python scripts/issue541_prescreen.py --step 0d "${SMOKE_FLAG[@]}"

# ── G0 gate (plan §7): deterministic branch read from prior_screen.json ─────
if [[ "$DRYRUN" == "1" ]]; then
  GATE_BRANCH="GO-full(dryrun)"
else
  GATE_BRANCH=$(uv run python -c "
import json; print(json.load(open('$SCREEN_JSON'))['gate']['branch'])")
fi
echo "[gate] G0 branch: $GATE_BRANCH"
if [[ "$GATE_BRANCH" == "NO-GO" && "$SMOKE" != "1" ]]; then
  # Train NOTHING; the Phase-0 write-up IS the deliverable (plan §7).
  log_phase results_sentinel
  runpy uv run python scripts/run_experiment_541.py --arm marine_biologist --phase results-sentinel
  log_phase done
  exit 0
fi

# Arms for this run (smoke: forced to marine; full: prior_screen selection).
if [[ "$SMOKE" == "1" ]]; then
  ARMS=(marine_biologist)
  SEEDS=(42)
else
  mapfile -t ARMS < <(uv run python -c "
import json
sel = json.load(open('$SCREEN_JSON'))['selection']
print('\n'.join(sel['sources']))")
  SEEDS=(42 137 256)
fi
echo "[arms] ${ARMS[*]} | seeds: ${SEEDS[*]}"

PANEL_CSV=$(uv run python -c "
import json, os
if os.environ.get('EPM_541_SMOKE') == '1':
    print('marine_biologist,local_historian,assistant')
else:
    print(','.join(json.load(open('$SCREEN_JSON'))['selection']['panel']))")

arm_slug() {
  uv run python -c "
import json, os
if os.environ.get('EPM_541_SMOKE') == '1' and '$1' == 'marine_biologist':
    print('arm_marine_biologist'); raise SystemExit
print(json.load(open('$SCREEN_JSON'))['selection']['arm_slugs']['$1'])"
}

# ── dataset (per arm, parallel <= N_GPUS; vLLM on-policy negatives) ─────────
log_phase dataset
for arm in "${ARMS[@]}"; do
  echo "runpy uv run python scripts/run_experiment_541.py --arm $arm --phase dataset --gpu-id {GPU} ${SMOKE_FLAG[*]}"
done | run_wave

# ── baselines: ONE shared 24-panel run + auto-chained 5-way judge ───────────
log_phase baselines
runpy uv run python scripts/run_experiment_541.py --arm marine_biologist --phase baselines --gpu-id 0 "${SMOKE_FLAG[@]}"

# ── training: all (arm × seed) cells, <= N_GPUS at a time ───────────────────
log_phase train
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "runpy uv run python scripts/run_experiment_541.py --arm $arm --phase worker --condition on-policy-suppression-cn --seed $seed --gpu-id {GPU} ${SMOKE_FLAG[*]}"
  done
done | run_wave

# ── trained eval: per arm (3 cells sequential inside) + auto 5-way rejudge ──
log_phase eval
for arm in "${ARMS[@]}"; do
  echo "runpy uv run python scripts/run_experiment_541.py --arm $arm --phase full-eval --gpu-id {GPU} ${SMOKE_FLAG[*]}"
done | run_wave

# ── engagement covariates: PRE-treatment (primary) + post-treatment texture ─
log_phase engagement
runpy uv run python scripts/issue541_engagement.py --pass base "${SMOKE_FLAG[@]}"
runpy uv run python scripts/issue541_engagement.py --pass trained "${SMOKE_FLAG[@]}"

# ── aggregate: parent rollup + #500-cleaned aggregator + predictors + plots ─
log_phase aggregate
for arm in "${ARMS[@]}"; do
  slug=$(arm_slug "$arm")
  runpy uv run python scripts/run_experiment_541.py --arm "$arm" --phase aggregate --gpu-id 0 "${SMOKE_FLAG[@]}"
  runpy uv run python scripts/aggregate_issue500.py --arm "$arm" --arm-slug "$slug" \
    --eval-root "$EVAL_ROOT_NAME" --panel "$PANEL_CSV"
done
runpy uv run python scripts/issue541_predictors.py
runpy uv run python scripts/plot_issue541.py

# ── upload: raw completions to the HF data bucket (skipped in smoke) ────────
if [[ "$SMOKE" != "1" ]]; then
  log_phase upload
  for arm in "${ARMS[@]}"; do
    runpy uv run python scripts/run_experiment_541.py --arm "$arm" --phase upload --gpu-id 0
  done
fi

# ── results sentinel, then the terminal phase marker ────────────────────────
log_phase results_sentinel
runpy uv run python scripts/run_experiment_541.py --arm "${ARMS[0]}" --phase results-sentinel "${SMOKE_FLAG[@]}"

log_phase done
