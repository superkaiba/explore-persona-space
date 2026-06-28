#!/usr/bin/env bash
# Issue #685 dispatch — orchestrate Phase A -> B.2 -> B -> C -> D in sequence.
#
# UNIFIED end-to-end: this script runs the SAME python entrypoint chain in both
# real and --smoke modes; --smoke just threads through to each phase (tiny slice,
# *_smoke output roots). No separate smoke/sweep code path.
#
#   bash scripts/issue685_dispatch.sh                       # full run (geometry + validity)
#   bash scripts/issue685_dispatch.sh --smoke               # tiny CPU verification (no pod, no GPU)
#   bash scripts/issue685_dispatch.sh --followup            # v3 follow-up chain (geometry FROZEN)
#   bash scripts/issue685_dispatch.sh --smoke --followup    # tiny CPU verification of the follow-up
#
# Phases (full / non-followup):
#   A    extract context vectors v_l(C), v_l(C+b)  -> store/issue685[_smoke]/*.pt   (GPU)
#   B.2  known behavior directions u_l(b)          -> store/.../instruct_known_directions.pt (GPU)
#   B    geometry metrics + null + projection      -> eval_results/.../metrics.json (CPU)
#   C    behavioral-validity judge (10 contexts)   -> eval_results/.../validity_judged.json (GPU+judge)
#   D    figures                                   -> figures/issue_685[_smoke]/   (CPU)
#   upload  Phase-C completions + analysis tensors -> HF data repo (real run only)
#
# Follow-up chain (--followup; v3 amendment, geometry FROZEN / not re-run):
#   C    full 10-context manipulation check (reuse-merges the 4 parent contexts)
#   C'   sycophancy on the opinion-bearing bank   -> validity_judged_syco_opinion.json
#   D'   refresh validity bars only (--validity-only; frozen geometry figures untouched)
#   upload  extended/new validity JSONs + raw gens -> HF data repo (real run only)

set -euo pipefail

# REPO_ROOT defaults to the GCP workload root or the RunPod path; honored if
# pre-exported by the dispatch (#641: GCE startup exports it).
REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
if [ ! -d "$REPO_ROOT/scripts" ]; then
  # Fall back to the dir this script lives in (local VM smoke).
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

SMOKE=""
FOLLOWUP=""
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE="--smoke" ;;
    --followup) FOLLOWUP="1" ;;
  esac
done

# Auto-detect the follow-up regime when the frozen geometry artifacts already
# exist (so a bare `bash issue685_dispatch.sh` on a synced tree re-folds the
# validity arms instead of re-running geometry). --followup forces it on.
if [ -z "$FOLLOWUP" ] && [ -z "$SMOKE" ] && [ -f "eval_results/issue_685/metrics.json" ] \
    && [ -f "store/issue685/instruct_context_vectors.pt" ]; then
  echo "[issue685.dispatch] frozen geometry artifacts present -> follow-up chain (geometry skipped)."
  FOLLOWUP="1"
fi

# Load credentials for the judge / HF (never bare load_dotenv in a heredoc;
# set -a && source .env && set +a per research-project-structure.md).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "[issue685.dispatch] REPO_ROOT=$REPO_ROOT SMOKE='${SMOKE}' FOLLOWUP='${FOLLOWUP}'"

if [ -z "$FOLLOWUP" ]; then
  # Geometry phases (frozen + skipped under --followup).
  echo "[phase=a_extract]"
  uv run python scripts/issue685_extract_shifts.py $SMOKE

  echo "[phase=b2_known_directions]"
  uv run python scripts/issue685_known_directions.py $SMOKE

  echo "[phase=b_metrics]"
  uv run python scripts/issue685_compute_metrics.py $SMOKE
fi

echo "[phase=c_judge_validity]"
uv run python scripts/issue685_judge_validity.py $SMOKE

if [ -n "$FOLLOWUP" ]; then
  echo "[phase=c_judge_syco_opinion]"
  uv run python scripts/issue685_judge_syco_opinion.py $SMOKE

  echo "[phase=d_figures]"
  uv run python scripts/issue685_make_figures.py $SMOKE --validity-only
else
  echo "[phase=d_figures]"
  uv run python scripts/issue685_make_figures.py $SMOKE
fi

if [ -z "$SMOKE" ]; then
  echo "[phase=upload]"
  uv run python scripts/issue685_upload.py
fi

echo "[phase=done]"
