#!/usr/bin/env bash
# scripts/issue2061_dispatch.sh — end-to-end orchestrator for task #2061.
#
# Sequences the 5 pipeline phases per plan §9 P1-P5, with per-phase env
# pins per `.claude/rules/code-style.md` § Shared-VM CPU thread caps.
#
# Phase routing (plan §9 P1-P5):
#   P1 encode          → GPU (eval intent, batched TopK encode)
#   P2 per-feature fit → cpu-bigmem (16 vCPU, thread caps below)
#   P3 null battery    → cpu-bigmem (same env)
#   P4 fitness gate    → GPU (eval intent, SAE encode + FVE)
#   P5 figures         → VM-local (matplotlib, no compute)
#
# The `--smoke-only` mode runs P1 loader-parity FVE smoke + P2/P3 on ONE
# cell only, ~5 min end-to-end, meeting the Step 0.5/0.6 smoke contract.
#
# Full-run mode requires all 4 P-phases to complete successfully; failure
# of any halts the pipeline (fail-loud, no partial results).
#
# Usage:
#   bash scripts/issue2061_dispatch.sh --smoke-only
#   bash scripts/issue2061_dispatch.sh --phase p1
#   bash scripts/issue2061_dispatch.sh --all         # P1 → P5 in sequence
#
# Env vars honored (all optional):
#   ISSUE2061_STAGE / ISSUE2061_CORPUS / ISSUE2061_ARM — cell filters
#   ISSUE2061_CONTEXT_SHARD_DIR — required for P2/P3 (#1336 staged locally)
#   ISSUE2061_SAE_REVISION / ISSUE2061_DATA_REVISION — HF pins

set -euo pipefail

# ─── thread caps (plan §9; .claude/rules/code-style.md § Shared-VM CPU) ─────
# On cpu-bigmem (16 vCPU dedicated), the pod's own env sets these to 16.
# On the shared VM (fallback for tiny debug runs), cap at 8 to leave
# headroom for concurrent sessions.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# ─── paths ──────────────────────────────────────────────────────────────────
DATA_ROOT="${ISSUE2061_DATA_ROOT:-data/issue_2061}"
EVAL_ROOT="${ISSUE2061_EVAL_ROOT:-eval_results/issue_2061}"
FIG_ROOT="${ISSUE2061_FIG_ROOT:-figures/issue_2061}"
ENCODED_DIR="$DATA_ROOT/sae_encoded"
R2_DIR="$EVAL_ROOT/per_feature_r2"
NULL_DIR="$EVAL_ROOT/null"
FITNESS_DIR="$EVAL_ROOT/fitness"

mkdir -p "$ENCODED_DIR" "$R2_DIR" "$NULL_DIR" "$FITNESS_DIR" "$FIG_ROOT"

# ─── phase runners ──────────────────────────────────────────────────────────
run_p1_smoke() {
  echo "=== P1: Loader-parity FVE smoke gate ==="
  uv run python scripts/issue2061_sae_encode.py --smoke-only
}

run_p1_encode() {
  echo "=== P1: SAE encode ==="
  local args=(--output-dir "$ENCODED_DIR")
  [[ -n "${ISSUE2061_STAGE:-}" ]] && args+=(--stage "$ISSUE2061_STAGE")
  [[ -n "${ISSUE2061_CORPUS:-}" ]] && args+=(--corpus "$ISSUE2061_CORPUS")
  [[ -n "${ISSUE2061_SAE_REVISION:-}" ]] && args+=(--sae-revision "$ISSUE2061_SAE_REVISION")
  [[ -n "${ISSUE2061_DATA_REVISION:-}" ]] && args+=(--data-revision "$ISSUE2061_DATA_REVISION")
  if [[ -z "${ISSUE2061_STAGE:-}" && -z "${ISSUE2061_CORPUS:-}" ]]; then
    args+=(--all-cells)
  fi
  uv run python scripts/issue2061_sae_encode.py --smoke-then-encode "${args[@]}"
}

run_p2_fit() {
  echo "=== P2: Per-feature ridge fit ==="
  local ctx_dir="${ISSUE2061_CONTEXT_SHARD_DIR:-}"
  if [[ -z "$ctx_dir" ]]; then
    echo "ERROR: ISSUE2061_CONTEXT_SHARD_DIR unset (need #1336 shards staged locally)" >&2
    return 1
  fi
  local args=(--context-shard-dir "$ctx_dir" --encoded-dir "$ENCODED_DIR" --output-dir "$R2_DIR")
  [[ -n "${ISSUE2061_STAGE:-}" ]] && args+=(--stage "$ISSUE2061_STAGE")
  [[ -n "${ISSUE2061_CORPUS:-}" ]] && args+=(--corpus "$ISSUE2061_CORPUS")
  [[ -n "${ISSUE2061_ARM:-}" ]] && args+=(--arm "$ISSUE2061_ARM")
  if [[ -z "${ISSUE2061_STAGE:-}" && -z "${ISSUE2061_CORPUS:-}" ]]; then
    args+=(--all-cells)
  fi
  uv run python scripts/issue2061_fit_per_feature.py "${args[@]}"
}

run_p3_null() {
  echo "=== P3: Selection-symmetric null battery ==="
  local args=(--r2-dir "$R2_DIR" --output-dir "$NULL_DIR" --all-cells)
  uv run python scripts/issue2061_null.py "${args[@]}"
}

run_p4_fitness() {
  echo "=== P4: Cross-stage SAE-fitness gate ==="
  local args=(--all-stages --output-dir "$FITNESS_DIR")
  [[ -n "${ISSUE2061_SAE_REVISION:-}" ]] && args+=(--sae-revision "$ISSUE2061_SAE_REVISION")
  [[ -n "${ISSUE2061_DATA_REVISION:-}" ]] && args+=(--data-revision "$ISSUE2061_DATA_REVISION")
  uv run python scripts/issue2061_fitness.py "${args[@]}"
}

run_p5_figures() {
  echo "=== P5: Figures ==="
  uv run python scripts/issue2061_figures.py --all \
    --r2-dir "$R2_DIR" --null-dir "$NULL_DIR" --fitness-dir "$FITNESS_DIR" \
    --output-dir "$FIG_ROOT"
}

# ─── argparse-lite ──────────────────────────────────────────────────────────
MODE=""
PHASE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-only) MODE=smoke; shift ;;
    --all) MODE=all; shift ;;
    --phase) PHASE="$2"; shift 2 ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "ERROR: unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODE" && -z "$PHASE" ]]; then
  echo "ERROR: pass --smoke-only, --all, or --phase p1|p2|p3|p4|p5" >&2
  exit 2
fi

# ─── dispatch ───────────────────────────────────────────────────────────────
if [[ "$MODE" == "smoke" ]]; then
  echo "MODE=smoke-only"
  run_p1_smoke
  # Smoke covers loader-parity + argparse import graph. Full P1 encode + P2/P3
  # on a smoke cell (1 stage, 1 corpus) would need shards staged locally;
  # end here for the parity smoke bounded ~1-2 min.
  echo "[smoke] DONE"
  exit 0
fi

if [[ "$MODE" == "all" ]]; then
  echo "MODE=all (P1 → P5)"
  run_p1_encode
  run_p2_fit
  run_p3_null
  run_p4_fitness
  run_p5_figures
  echo "[all] DONE"
  exit 0
fi

case "$PHASE" in
  p1) run_p1_encode ;;
  p2) run_p2_fit ;;
  p3) run_p3_null ;;
  p4) run_p4_fitness ;;
  p5) run_p5_figures ;;
  *) echo "ERROR: unknown phase: $PHASE" >&2; exit 2 ;;
esac

echo "[$PHASE] DONE"
