#!/usr/bin/env bash
# #1434 pod-side dispatcher (plan §10 workload command).
#
# Sequences the GPU phases on the provisioned pod/instance:
#   dispatch (12-run train/ladder/tier2/margin fan-out; work-conserving,
#   CVD pinned per slot) -> base-arms -> panel -> pv extract -> pv project.
# The VM phases (datagen/stage before; judge-analyze/validate after) run
# off-pod. `[phase=done]` is emitted HERE only (the fu3/fu4 convention);
# pod-side code never shells scripts/task.py (sentinel-file contract).
#
# Usage: bash scripts/issue1434_dispatch.sh --phase pod-all [--smoke] \
#          [--out-root PATH] [--manifest PATH] [--cells ws-pers,...]
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

PHASE="pod-all"
MODE="--full"
OUT_ROOT="data/issue_1434/cells"
SENTINEL_DIR="${SENTINEL_DIR:-/workspace/logs}"
MANIFEST="eval_results/issue_1434/cell_manifest_i1434.json"
CELLS=""
RUNS=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --smoke) MODE="--smoke"; shift ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --sentinel-dir) SENTINEL_DIR="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --cells) CELLS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
if [ "$MODE" = "--smoke" ] && [ "$OUT_ROOT" = "data/issue_1434/cells" ]; then
  OUT_ROOT="/tmp/issue-1434-i1434-smoke"   # scratch redirect: smoke never
  MANIFEST="$OUT_ROOT/cell_manifest_i1434.json"  # touches committed paths
fi
mkdir -p "$SENTINEL_DIR" "$OUT_ROOT"

WORKER="scripts/issue1434_worker.py"
PV="scripts/issue1434_pv.py"
COMMON=("$MODE" --out-root "$OUT_ROOT" --sentinel-dir "$SENTINEL_DIR")
[ -n "$CELLS" ] && COMMON+=(--cells "$CELLS")
export WANDB_PROJECT="${WANDB_PROJECT:-issue1434}"

run_phase() {
  echo "[issue1434-dispatch] >>> $*"
  uv run python "$@"
}

case "$PHASE" in
  pod-all)
    # 12-run fan-out: fu4's work-conserving dispatcher (CVD pinned per slot,
    # width = detect_n_gpus — never narrowed under --smoke).
    DISPATCH_ARGS=("$WORKER" "$MODE" --phase dispatch
                   --out-root "$OUT_ROOT" --sentinel-dir "$SENTINEL_DIR")
    [ -n "$RUNS" ] && DISPATCH_ARGS+=(--runs "$RUNS")
    if [ "$MODE" = "--smoke" ]; then
      # Parent smoke convention: the K1 pin verifies the FIXTURE's own sha
      # (fu2.build_smoke_mix_fixture); the manifest pin is the FULL-run gate.
      DISPATCH_ARGS+=(--no-upload)
    else
      DISPATCH_ARGS+=(--manifest "$MANIFEST")
    fi
    run_phase "${DISPATCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
    BA_ARGS=("$WORKER" "$MODE" --phase base-arms "${COMMON[@]:1}")
    PN_ARGS=("$WORKER" "$MODE" --phase panel "${COMMON[@]:1}")
    EX_ARGS=("$PV" "$MODE" --phase extract "${COMMON[@]:1}")
    PJ_ARGS=("$PV" "$MODE" --phase project "${COMMON[@]:1}")
    if [ "$MODE" = "--smoke" ]; then
      BA_ARGS+=(--no-upload); PN_ARGS+=(--no-upload)
      EX_ARGS+=(--no-upload); PJ_ARGS+=(--no-upload)
    fi
    run_phase "${BA_ARGS[@]}"
    run_phase "${PN_ARGS[@]}"
    run_phase "${EX_ARGS[@]}"
    run_phase "${PJ_ARGS[@]}"
    ;;
  *)
    # Single-phase passthrough (crash-fix / resume surface).
    run_phase "$WORKER" "$MODE" --phase "$PHASE" "${COMMON[@]:1}" "${EXTRA_ARGS[@]}"
    ;;
esac

echo "[phase=done]"
