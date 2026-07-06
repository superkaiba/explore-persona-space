#!/usr/bin/env bash
# Issue #1073 phase driver: P0 probe -> P1 greedy gen -> P2 stoch10 gen ->
# P3 capture -> P4 fits (-> optional P5 figures with --with-figures).
#
# Pod/GCE-safe: REPO_ROOT honors the GCE lane's exported $WORKLOAD_ROOT
# (gotchas #599/#641), .env is sourced CONDITIONALLY (the GCE lane exports
# tokens via instance metadata and has NO .env — gotchas #923). The single
# terminal `[phase=done]` line is emitted HERE (reserved token; phase scripts
# emit only their own `[phase=pN]` breadcrumbs); the results sentinel is
# written by issue1073_fits.py BEFORE that line (pod-side-reporting contract).
#
# Usage:
#   bash scripts/issue1073_driver.sh --all [--smoke] [--no-upload] [...]
#   bash scripts/issue1073_driver.sh --from-phase p3
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ALL_PHASES=(p0 p1 p2 p3 p4)
FROM=""
RUN_ALL=0
WITH_FIGURES=0
SMOKE_FLAG=()
NOUPLOAD_FLAG=()
OUTROOT_ARGS=()
MODEL_ARGS=()
PILOT_ARGS=()
NBOOT_ARGS=()
NFOLDS_ARGS=()
DEVICE_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --all) RUN_ALL=1 ;;
    --from-phase) FROM="$2"; shift ;;
    --with-figures) WITH_FIGURES=1 ;;
    --smoke) SMOKE_FLAG=(--smoke) ;;
    --no-upload) NOUPLOAD_FLAG=(--no-upload) ;;
    --out-root) OUTROOT_ARGS=(--out-root "$2"); shift ;;
    --model) MODEL_ARGS=(--model "$2"); shift ;;
    --pilot-n) PILOT_ARGS=(--pilot-n "$2"); shift ;;
    --n-boot) NBOOT_ARGS=(--n-boot "$2"); shift ;;
    --n-folds) NFOLDS_ARGS=(--n-folds "$2"); shift ;;
    --device) DEVICE_ARGS=(--device "$2"); shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$RUN_ALL" -eq 0 ] && [ -z "$FROM" ]; then
  echo "pass --all or --from-phase pN" >&2
  exit 2
fi

PHASES=()
STARTED=0
for ph in "${ALL_PHASES[@]}"; do
  if [ "$RUN_ALL" -eq 1 ] || [ "$ph" = "$FROM" ]; then STARTED=1; fi
  if [ "$STARTED" -eq 1 ]; then PHASES+=("$ph"); fi
done
if [ "${#PHASES[@]}" -eq 0 ]; then
  echo "unknown --from-phase '$FROM' (expected one of: ${ALL_PHASES[*]})" >&2
  exit 2
fi
echo "[driver] phases: ${PHASES[*]} (figures: $WITH_FIGURES)"

run_phase() {
  case "$1" in
    p0)
      uv run python scripts/issue1073_gen.py --phase p0 \
        ${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"} ${NOUPLOAD_FLAG[@]+"${NOUPLOAD_FLAG[@]}"} \
        ${OUTROOT_ARGS[@]+"${OUTROOT_ARGS[@]}"} ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} \
        ${PILOT_ARGS[@]+"${PILOT_ARGS[@]}"}
      ;;
    p1|p2)
      uv run python scripts/issue1073_gen.py --phase "$1" \
        ${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"} ${NOUPLOAD_FLAG[@]+"${NOUPLOAD_FLAG[@]}"} \
        ${OUTROOT_ARGS[@]+"${OUTROOT_ARGS[@]}"} ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
      ;;
    p3)
      uv run python scripts/issue1073_capture.py \
        ${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"} ${NOUPLOAD_FLAG[@]+"${NOUPLOAD_FLAG[@]}"} \
        ${OUTROOT_ARGS[@]+"${OUTROOT_ARGS[@]}"} ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
      ;;
    p4)
      uv run python scripts/issue1073_fits.py \
        ${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"} ${NOUPLOAD_FLAG[@]+"${NOUPLOAD_FLAG[@]}"} \
        ${OUTROOT_ARGS[@]+"${OUTROOT_ARGS[@]}"} ${NBOOT_ARGS[@]+"${NBOOT_ARGS[@]}"} \
        ${NFOLDS_ARGS[@]+"${NFOLDS_ARGS[@]}"} ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"}
      ;;
    *)
      echo "unknown phase: $1" >&2
      exit 2
      ;;
  esac
}

for ph in "${PHASES[@]}"; do
  echo "[driver] running phase $ph"
  run_phase "$ph"
done

if [ "$WITH_FIGURES" -eq 1 ]; then
  uv run python scripts/issue1073_figures.py \
    ${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"} ${OUTROOT_ARGS[@]+"${OUTROOT_ARGS[@]}"}
fi

echo "[phase=done]"
