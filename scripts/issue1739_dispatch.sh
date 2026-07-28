#!/usr/bin/env bash
# Issue #1739 dispatcher frame (round A).
#
# Phases: gates | extract | capture | judge | fits | figures
#   --phase <p>       run exactly one phase
#   --from-phase <p>  run <p> and every later phase
# Round A implements ONLY the `gates` phase; later phases exit 3 with a
# round-B/C note (and still write their sentinel so the poller sees them).
#
# Pod-side signaling is by SENTINEL FILE ONLY
# (${OUT_ROOT:-/workspace/logs}/issue-1739-<phase>.json) — NEVER a
# scripts/task.py shellout from pod-side code (hard project rule; the VM
# poller drains sentinels into markers).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

OUT_ROOT="${OUT_ROOT:-/workspace/logs}"
mkdir -p "$OUT_ROOT"

PHASES=(gates extract capture judge fits figures)

usage() {
  cat <<'EOF'
Usage: bash scripts/issue1739_dispatch.sh [--phase <p>] [--from-phase <p>]
Phases: gates extract capture judge fits figures
Round A: only `gates` is implemented; later phases exit 3 (round B/C).
Env: OUT_ROOT (sentinel dir; default /workspace/logs), REPO_ROOT.
EOF
}

# Shared-VM thread caps on VM-side python only (pods/GCE keep full width).
CAPS=()
if [ -d /mnt/eps-data ] || [ "$(hostname 2>/dev/null || true)" = "cia-benchmark-vm" ]; then
  CAPS=(env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2)
fi

write_sentinel() {
  # write_sentinel <phase> <status> <rc>
  local phase="$1" status="$2" rc="$3" ts commit
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  printf '{"issue": 1739, "phase": "%s", "status": "%s", "rc": %s, "ts": "%s", "git_commit": "%s"}\n' \
    "$phase" "$status" "$rc" "$ts" "$commit" > "${OUT_ROOT}/issue-1739-${phase}.json"
}

# Behaviors + optional smoke slice (empty EPM_I1739_LIMIT = production, no cap).
BEHAVIORS_RUN="${EPM_I1739_BEHAVIORS:-evil sycophancy hallucination}"
LIMIT_ARGS=()
if [ -n "${EPM_I1739_LIMIT:-}" ]; then LIMIT_ARGS=(--limit "$EPM_I1739_LIMIT"); fi
CTX_LIMIT_ARGS=()
if [ -n "${EPM_I1739_LIMIT:-}" ]; then CTX_LIMIT_ARGS=(--max-contexts "$EPM_I1739_LIMIT"); fi
# E1 extraction has no context cap (5 pairs x 2 signs x 20 questions is fixed);
# the smoke slice narrows ROLLOUTS per job only. Production default (no
# EPM_I1739_LIMIT) keeps the full E1_N_ROLLOUTS=10.
E1_LIMIT_ARGS=()
if [ -n "${EPM_I1739_LIMIT:-}" ]; then E1_LIMIT_ARGS=(--n-rollouts 2); fi

run_phase() {
  local phase="$1" b
  echo "[phase=${phase}] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  case "$phase" in
    gates)
      "${CAPS[@]}" uv run python scripts/issue1739_gates.py --gate all
      write_sentinel "$phase" ok 0
      echo "[phase=${phase}] done"
      ;;
    extract)
      # Staging (streaming HF loads, checkpointed/resumable) + labeling
      # generation (K rollouts/context) + E1 extraction generation.
      for b in $BEHAVIORS_RUN; do
        echo "[phase=${phase}] staging behavior=${b}"
        "${CAPS[@]}" uv run python -c "
import sys
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.experiments.issue_1739.corpus_staging import stage_corpus
b = sys.argv[1]
cap = int(sys.argv[2]) if sys.argv[2] != 'none' else None
stage_corpus(b, 'train', cap, 0)
stage_corpus(b, 'eval', cap, 0)
" "$b" "${EPM_I1739_LIMIT:-none}"
        echo "[phase=${phase}] labeling generation behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_generate.py --mode labeling \
          --behavior "$b" \
          --contexts-jsonl data/issue_1739/staged/"$b"/"$b"_*_*.contexts.jsonl \
          --out-root raw_completions/issue_1739 "${CTX_LIMIT_ARGS[@]}"
        echo "[phase=${phase}] E1 extraction generation behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_generate.py --mode extraction \
          --behavior "$b" --out-root raw_completions/issue_1739 \
          --inputs-dir data/issue_1739/inputs "${E1_LIMIT_ARGS[@]}"
      done
      write_sentinel "$phase" ok 0
      echo "[phase=${phase}] done (extract)"
      ;;
    capture)
      for b in $BEHAVIORS_RUN; do
        echo "[phase=${phase}] capture behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_capture.py \
          --rollout-dir raw_completions/issue_1739/labeling/"$b" \
          --store-dir data/issue_1739/store/"$b"_labeling "${LIMIT_ARGS[@]}"
        echo "[phase=${phase}] E1 extraction capture behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_capture.py \
          --rollout-dir raw_completions/issue_1739/extraction/"$b" \
          --store-dir data/issue_1739/store/"$b"_extraction "${LIMIT_ARGS[@]}"
      done
      write_sentinel "$phase" ok 0
      echo "[phase=${phase}] done (capture)"
      ;;
    judge)
      for b in $BEHAVIORS_RUN; do
        echo "[phase=${phase}] judge behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_judge.py \
          --behavior "$b" \
          --rollout-dir raw_completions/issue_1739/labeling/"$b" \
          --out-dir eval_results/issue_1739/judge/"$b" \
          --dv-out-root eval_results/issue_1739 "${LIMIT_ARGS[@]}"
      done
      write_sentinel "$phase" ok 0
      echo "[phase=${phase}] done (judge)"
      ;;
    fits)
      # Matched-budget arm grid per behavior (both prefix+context variants).
      # The smoke slice threads through EPM_I1739_LIMIT exactly like the
      # earlier phases (smoke IS the production path with tiny caps).
      for b in $BEHAVIORS_RUN; do
        echo "[phase=${phase}] fits behavior=${b}"
        FITS_ARGS=(--behavior "$b"
          --labeled-store data/issue_1739/store/"$b"_labeling
          --dv-json eval_results/issue_1739/dv_dataset/"$b"/labeling.json
          --u-store data/issue_1739/hf_dl/u_store
          --e1-store data/issue_1739/store/"$b"_extraction
          --out-root eval_results/issue_1739/"$b")
        if [ -n "${EPM_I1739_LIMIT:-}" ]; then
          FITS_ARGS+=(--budgets "$EPM_I1739_LIMIT" --u-size 64 --layers 0 1 2
            --n-boot 50 --n-perm 50 --mlp-epochs 5)
        fi
        "${CAPS[@]}" uv run python scripts/issue1739_fits.py "${FITS_ARGS[@]}"
      done
      write_sentinel "$phase" ok 0
      echo "[phase=${phase}] done (fits)"
      ;;
    figures)
      for b in $BEHAVIORS_RUN; do
        echo "[phase=${phase}] figures behavior=${b}"
        "${CAPS[@]}" uv run python scripts/issue1739_figures.py \
          --summary eval_results/issue_1739/"$b"/arm_results/all_arms_spearman.json \
          --out-dir figures/issue_1739/"$b"
      done
      write_sentinel "$phase" ok 0
      echo "[phase=${phase}] done (figures)"
      ;;
    *)
      echo "unknown phase: ${phase}" >&2
      return 2
      ;;
  esac
}

PHASE=""
FROM_PHASE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="${2:?--phase needs a value}"; shift 2 ;;
    --from-phase) FROM_PHASE="${2:?--from-phase needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

valid_phase() {
  local p
  for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done
  return 1
}

if [ -n "$PHASE" ] && [ -n "$FROM_PHASE" ]; then
  echo "--phase and --from-phase are mutually exclusive" >&2
  exit 2
fi
if [ -n "$PHASE" ]; then
  valid_phase "$PHASE" || { echo "unknown phase: $PHASE" >&2; exit 2; }
  run_phase "$PHASE"
  exit $?
fi

START="${FROM_PHASE:-gates}"
valid_phase "$START" || { echo "unknown phase: $START" >&2; exit 2; }
started=0
for p in "${PHASES[@]}"; do
  [ "$p" = "$START" ] && started=1
  [ "$started" = 1 ] || continue
  run_phase "$p"
done
