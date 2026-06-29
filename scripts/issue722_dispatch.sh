#!/usr/bin/env bash
# Issue #722 — function-change (M0 vs M⁺) measurement driver.
#
# ONE entrypoint for BOTH the smoke and the production sweep (smoke = the sweep
# with --smoke flags scaling each phase to one cell). No per-cell subprocess
# fan-out: each phase is a single in-process Python call, so the smoke exercises
# the IDENTICAL dispatcher the sweep runs (PASS_UNIFIED — see the
# epm:smoke-architecture-check marker).
#
# Phases (all on the GCP eval-h100 lane, or a local CPU smoke):
#   1. issue722_extract_fact_rb.py  — re-extract the taught-fact r_B (HF upload)
#   2. issue722_fit_M.py            — fit M0/M⁺/M_pseudo + the four reads → cells/
#   3. issue722_analyze.py          — assemble cells/ → the 4 deliverable JSONs
#
# The GCP lane's startup script owns the EXIT-trap that persists
# eval_results/issue_722/ to HF under issue722_function_change/<attempt_id>/ on
# both success and crash; this script only needs to produce the eval JSONs in
# eval_results/issue_722/.
#
# Usage:
#   bash scripts/issue722_dispatch.sh                 # full production sweep
#   bash scripts/issue722_dispatch.sh --smoke         # tiny end-to-end smoke
#   bash scripts/issue722_dispatch.sh --resume-from-attempt att-20260628-235255
#                                                     # re-launch: stage the 3 clean
#                                                     # em cells from the crashed
#                                                     # attempt + skip fact_rb_extract
#   bash scripts/issue722_dispatch.sh --skip-fact-rb-extract   # r_b_fact.pt already on HF
#   EPM_FACT_DEVICE=cpu EPM_FACT_MODEL=<tiny> bash scripts/issue722_dispatch.sh --smoke
set -euo pipefail

REPO_ROOT="${WORKLOAD_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Cross-backend convention: name the WandB project before any wandb call.
export WANDB_PROJECT="${WANDB_PROJECT:-issue722}"

# Parse flags (order-independent): --smoke, --skip-fact-rb-extract,
# --resume-from-attempt <id>. The resume flag IMPLIES skipping fact_rb_extract
# (r_b_fact.pt is durable on HF from the crashed attempt; re-extracting wastes
# the ~3 min GPU forward pass), so a re-launch only needs the one flag.
SMOKE=""
SKIP_FACT=""
RESUME_ATTEMPT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE="--smoke"; shift ;;
    --skip-fact-rb-extract) SKIP_FACT="1"; shift ;;
    --resume-from-attempt)
      RESUME_ATTEMPT="${2:?--resume-from-attempt needs an attempt id}"
      SKIP_FACT="1"
      shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Phase 1 — taught-fact r_B extraction. Device/model overridable for a CPU smoke.
# Skipped on a re-launch when r_b_fact.pt already exists on HF (--skip-fact-rb-extract
# or implied by --resume-from-attempt); the fit phase fail-loud-asserts the artifact.
if [ -n "$SKIP_FACT" ]; then
  echo "[phase=fact_rb_extract] skipped (--skip-fact-rb-extract; r_b_fact.pt durable on HF)"
else
  FACT_DEVICE="${EPM_FACT_DEVICE:-cuda}"
  FACT_MODEL="${EPM_FACT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
  FACT_ARGS="--device ${FACT_DEVICE} --model ${FACT_MODEL} --skip-if-exists"
  if [ -n "$SMOKE" ]; then
    FACT_ARGS="$FACT_ARGS --smoke"
  fi
  echo "[phase=fact_rb_extract] starting (device=${FACT_DEVICE} model=${FACT_MODEL})"
  # shellcheck disable=SC2086
  uv run python scripts/issue722_extract_fact_rb.py $FACT_ARGS
fi

# Phase 2 — fit M0 vs M⁺ + the four reads.
FIT_ARGS=""
if [ -n "$SMOKE" ]; then
  FIT_ARGS="--smoke"
fi
if [ -n "$RESUME_ATTEMPT" ]; then
  FIT_ARGS="$FIT_ARGS --resume-from-attempt $RESUME_ATTEMPT"
fi
echo "[phase=fit_M] starting"
# shellcheck disable=SC2086
uv run python scripts/issue722_fit_M.py $FIT_ARGS

# Phase 3 — assemble the 4 deliverable JSONs.
echo "[phase=analyze] starting"
uv run python scripts/issue722_analyze.py

echo "[phase=done] issue722 dispatch complete"
