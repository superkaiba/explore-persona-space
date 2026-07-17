#!/usr/bin/env bash
# #1434 pod-side dispatcher (plan §10 workload command).
#
# Work-conserving pod-all shape (plan §9: P7/P8 are dedicated streams; no GPU
# idles behind a stage barrier when an independent phase is pending):
#
#   t=0  ┬ dispatch (12-run train/ladder/tier2/margin fan-out; fu4
#        │   work-conserving queue on GPUs 0..N-3 via --n-gpus)
#        ├ pv extract   (base-model only — independent of training; GPU N-2)
#        └ base-arms    (base-model only — independent of training; GPU N-1)
#   join ┬ panel        (needs dispatch selections; GPU 0)
#        └ pv project   (needs dispatch selections + extract r_B; GPU 1)
#
# N>=3 GPUs: the shape above. N==2: dispatch on both GPUs first (a second
# vLLM engine cannot co-reside with a training run on one A100 — named
# capacity constraint), then extract∥base-arms, then panel∥project. N==1:
# fully serial (same constraint; nothing can overlap). N==0 (CPU smoke):
# the CONCURRENT shape, unpinned — the smoke exercises the production
# background/join path (no smoke-vs-production shape branch).
#
# The VM phases (datagen/stage before; judge-analyze/validate after) run
# off-pod. `[phase=done]` is emitted HERE only (the fu3/fu4 convention);
# pod-side code never shells scripts/task.py (sentinel-file contract).
# Background phase logs go to $OUT_ROOT/logs/phase_<name>.log; their
# completion echoes never carry the reserved `[phase=` token.
#
# Usage: bash scripts/issue1434_dispatch.sh --phase pod-all [--smoke] \
#          [--round i1434|i1434po] [--out-root PATH] [--manifest PATH] \
#          [--cells ws-pers,...]
#
# --round i1434po (plan §4 D2'-D4', the positive-only regime arm) runs the
# reduced chain: dispatch fan-out on EVERY GPU (no tail reservation — there
# is no concurrent extract/base-arms: r_B + base arms are REUSED from the
# parent at the pinned revision), then panel ∥ project.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

PHASE="pod-all"
MODE="--full"
ROUND="i1434"
OUT_ROOT="data/issue_1434/cells"
SENTINEL_DIR="${SENTINEL_DIR:-/workspace/logs}"
MANIFEST=""
CELLS=""
RUNS=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --smoke) MODE="--smoke"; shift ;;
    --round) ROUND="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --sentinel-dir) SENTINEL_DIR="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --cells) CELLS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
if [ -z "$MANIFEST" ]; then
  if [ "$ROUND" = "i1434po" ]; then
    MANIFEST="eval_results/issue_1434/writing-style-positive-only-regime/cell_manifest_i1434po.json"
  else
    MANIFEST="eval_results/issue_1434/cell_manifest_i1434.json"
  fi
fi
if [ "$MODE" = "--smoke" ] && [ "$OUT_ROOT" = "data/issue_1434/cells" ]; then
  OUT_ROOT="/tmp/issue-1434-${ROUND}-smoke"   # scratch redirect: smoke never
  MANIFEST="$OUT_ROOT/cell_manifest_${ROUND}.json"  # touches committed paths
fi
mkdir -p "$SENTINEL_DIR" "$OUT_ROOT"

WORKER="scripts/issue1434_worker.py"
PV="scripts/issue1434_pv.py"
COMMON=("$MODE" --round "$ROUND" --out-root "$OUT_ROOT" --sentinel-dir "$SENTINEL_DIR")
[ -n "$CELLS" ] && COMMON+=(--cells "$CELLS")
export WANDB_PROJECT="${WANDB_PROJECT:-issue1434}"

run_phase() {
  echo "[issue1434-dispatch] >>> $*"
  uv run python "$@"
}

# Reap still-running background phases on ANY exit (a foreground-phase crash
# under `set -e` must never orphan the concurrent tail phases).
BG_PIDS=""
trap '[ -n "$BG_PIDS" ] && kill $BG_PIDS 2>/dev/null || true' EXIT

# launch_bg <name> <cvd|-> <vllm_port|-> <python args...>
# Backgrounds one tail phase, CVD/port pinned (fu4 slot convention: physical
# GPU index + VLLM_PORT=8000+index so ports never clash with dispatch slots).
# Caller reads the pid from $! immediately after return.
launch_bg() {
  local name="$1" cvd="$2" port="$3"; shift 3
  local log="$OUT_ROOT/logs/phase_${name}.log"
  mkdir -p "$OUT_ROOT/logs"
  local -a envp=()
  [ "$cvd" != "-" ] && envp+=("CUDA_VISIBLE_DEVICES=${cvd}")
  [ "$port" != "-" ] && envp+=("VLLM_PORT=${port}")
  echo "[issue1434-dispatch] bg-launch ${name} (cvd=${cvd} port=${port} log=${log}) >>> $*"
  env ${envp[@]+"${envp[@]}"} uv run python "$@" > "$log" 2>&1 &
  BG_PIDS="$BG_PIDS $!"
}

# join_bg <name> <pid> — fail loud with the phase log tail on nonzero exit.
join_bg() {
  local name="$1" pid="$2"
  if ! wait "$pid"; then
    echo "[issue1434-dispatch] tail phase ${name} FAILED — last 60 log lines:"
    tail -60 "$OUT_ROOT/logs/phase_${name}.log" || true
    exit 1
  fi
  echo "[issue1434-dispatch] tail phase ${name} complete (log $OUT_ROOT/logs/phase_${name}.log)"
}

case "$PHASE" in
  pod-all)
    N_GPUS=$( (nvidia-smi -L 2>/dev/null || true) | grep -c '^GPU' || true )
    # 12-run fan-out: fu4's work-conserving dispatcher (CVD pinned per slot;
    # width never narrowed under --smoke — only by the 2-GPU tail reservation).
    DISPATCH_ARGS=("$WORKER" "$MODE" --round "$ROUND" --phase dispatch
                   --out-root "$OUT_ROOT" --sentinel-dir "$SENTINEL_DIR")
    [ -n "$RUNS" ] && DISPATCH_ARGS+=(--runs "$RUNS")
    if [ "$MODE" = "--smoke" ]; then
      # Parent smoke convention: the K1 pin verifies the FIXTURE's own sha
      # (fu2.build_smoke_mix_fixture); the manifest pin is the FULL-run gate.
      DISPATCH_ARGS+=(--no-upload)
    else
      DISPATCH_ARGS+=(--manifest "$MANIFEST")
    fi
    BA_ARGS=("$WORKER" "$MODE" --phase base-arms "${COMMON[@]:1}")
    PN_ARGS=("$WORKER" "$MODE" --phase panel "${COMMON[@]:1}")
    EX_ARGS=("$PV" "$MODE" --phase extract "${COMMON[@]:1}")
    PJ_ARGS=("$PV" "$MODE" --phase project "${COMMON[@]:1}")
    if [ "$MODE" = "--smoke" ]; then
      BA_ARGS+=(--no-upload); PN_ARGS+=(--no-upload)
      EX_ARGS+=(--no-upload); PJ_ARGS+=(--no-upload)
    fi
    if [ "$ROUND" = "i1434po" ]; then
      # po chain (plan §4 D2'-D4'): NO extract (r_B reused), NO base-arms
      # (base Tier-2 + panel row reused) — dispatch keeps EVERY GPU (no tail
      # reservation), then panel ∥ project on the freed GPUs.
      if [ "$N_GPUS" -eq 0 ]; then
        DISPATCH_ARGS+=(--n-gpus 1)   # CPU smoke: fu4 detect_n_gpus raises without nvidia-smi
        run_phase "${DISPATCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
        launch_bg panel - - "${PN_ARGS[@]}"; PN_PID=$!
        launch_bg project - - "${PJ_ARGS[@]}"; PJ_PID=$!
        join_bg panel "$PN_PID"
        join_bg project "$PJ_PID"
      elif [ "$N_GPUS" -eq 1 ]; then
        run_phase "${DISPATCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
        run_phase "${PN_ARGS[@]}"
        run_phase "${PJ_ARGS[@]}"
      else
        run_phase "${DISPATCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
        launch_bg panel 0 8000 "${PN_ARGS[@]}"; PN_PID=$!
        launch_bg project 1 8001 "${PJ_ARGS[@]}"; PJ_PID=$!
        join_bg panel "$PN_PID"
        join_bg project "$PJ_PID"
      fi
    elif [ "$N_GPUS" -ge 3 ]; then
      # Reserve the top 2 GPUs for the training-independent tail phases;
      # dispatch keeps slots 0..N-3 (12 runs stay 2 waves at any width >=6).
      DISPATCH_ARGS+=(--n-gpus "$((N_GPUS - 2))")
      launch_bg extract "$((N_GPUS - 2))" "$((8000 + N_GPUS - 2))" "${EX_ARGS[@]}"; EX_PID=$!
      launch_bg base_arms "$((N_GPUS - 1))" "$((8000 + N_GPUS - 1))" "${BA_ARGS[@]}"; BA_PID=$!
      run_phase "${DISPATCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
      join_bg extract "$EX_PID"
      join_bg base_arms "$BA_PID"
      launch_bg panel 0 8000 "${PN_ARGS[@]}"; PN_PID=$!
      launch_bg project 1 8001 "${PJ_ARGS[@]}"; PJ_PID=$!
      join_bg panel "$PN_PID"
      join_bg project "$PJ_PID"
    elif [ "$N_GPUS" -eq 0 ]; then
      # CPU smoke: SAME concurrent shape, unpinned (tiny stub models coexist).
      # fu4's detect_n_gpus() raises without nvidia-smi -> explicit 1-slot
      # width (a caller --n-gpus in EXTRA_ARGS still wins: argparse last-wins).
      DISPATCH_ARGS+=(--n-gpus 1)
      launch_bg extract - - "${EX_ARGS[@]}"; EX_PID=$!
      launch_bg base_arms - - "${BA_ARGS[@]}"; BA_PID=$!
      run_phase "${DISPATCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
      join_bg extract "$EX_PID"
      join_bg base_arms "$BA_PID"
      launch_bg panel - - "${PN_ARGS[@]}"; PN_PID=$!
      launch_bg project - - "${PJ_ARGS[@]}"; PJ_PID=$!
      join_bg panel "$PN_PID"
      join_bg project "$PJ_PID"
    elif [ "$N_GPUS" -eq 2 ]; then
      # Degraded width: dispatch saturates both GPUs first (a vLLM engine
      # cannot co-reside with a training run on one GPU — capacity
      # constraint), then the tail runs as two concurrent streams.
      run_phase "${DISPATCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
      launch_bg extract 0 8000 "${EX_ARGS[@]}"; EX_PID=$!
      launch_bg base_arms 1 8001 "${BA_ARGS[@]}"; BA_PID=$!
      join_bg extract "$EX_PID"
      join_bg base_arms "$BA_PID"
      launch_bg panel 0 8000 "${PN_ARGS[@]}"; PN_PID=$!
      launch_bg project 1 8001 "${PJ_ARGS[@]}"; PJ_PID=$!
      join_bg panel "$PN_PID"
      join_bg project "$PJ_PID"
    else
      # Single GPU: fully serial — nothing can overlap (same constraint).
      run_phase "${DISPATCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
      run_phase "${BA_ARGS[@]}"
      run_phase "${PN_ARGS[@]}"
      run_phase "${EX_ARGS[@]}"
      run_phase "${PJ_ARGS[@]}"
    fi
    ;;
  dose-select|dose-panel|dose-judge-analyze)
    # persona-dose-matched-regime phases (plan v8): same passthrough shape;
    # smoke adds --no-upload (pod-all convention — scratch out-root already
    # applies via the redirect above, so smoke never touches committed paths).
    DOSE_ARGS=("$WORKER" "$MODE" --phase "$PHASE" "${COMMON[@]:1}")
    [ "$MODE" = "--smoke" ] && DOSE_ARGS+=(--no-upload)
    run_phase "${DOSE_ARGS[@]}" "${EXTRA_ARGS[@]}"
    ;;
  *)
    # Single-phase passthrough (crash-fix / resume surface).
    run_phase "$WORKER" "$MODE" --phase "$PHASE" "${COMMON[@]:1}" "${EXTRA_ARGS[@]}"
    ;;
esac

echo "[phase=done]"
