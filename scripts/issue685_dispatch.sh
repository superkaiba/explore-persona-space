#!/usr/bin/env bash
# Issue #685 dispatch — orchestrate Phase A -> B.2 -> B -> C -> D in sequence.
#
# UNIFIED end-to-end: this script runs the SAME python entrypoint chain in both
# real and --smoke modes; --smoke just threads through to each phase (tiny slice,
# *_smoke output roots). No separate smoke/sweep code path.
#
#   bash scripts/issue685_dispatch.sh              # full run (GCP eval / RunPod H100)
#   bash scripts/issue685_dispatch.sh --smoke      # tiny CPU verification (no pod, no GPU)
#
# Phases:
#   A    extract context vectors v_l(C), v_l(C+b)  -> store/issue685[_smoke]/*.pt   (GPU)
#   B.2  known behavior directions u_l(b)          -> store/.../instruct_known_directions.pt (GPU)
#   B    geometry metrics + null + projection      -> eval_results/.../metrics.json (CPU)
#   C    behavioral-validity judge subset          -> eval_results/.../validity_judged.json (GPU+judge)
#   D    figures                                   -> figures/issue_685[_smoke]/   (CPU)
#   upload  Phase-C completions + analysis tensors -> HF data repo (real run only)

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
for arg in "$@"; do
  if [ "$arg" = "--smoke" ]; then
    SMOKE="--smoke"
  fi
done

# Load credentials for the judge / HF (never bare load_dotenv in a heredoc;
# set -a && source .env && set +a per research-project-structure.md).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "[issue685.dispatch] REPO_ROOT=$REPO_ROOT SMOKE='${SMOKE}'"
echo "[phase=a_extract]"
uv run python scripts/issue685_extract_shifts.py $SMOKE

echo "[phase=b2_known_directions]"
uv run python scripts/issue685_known_directions.py $SMOKE

echo "[phase=b_metrics]"
uv run python scripts/issue685_compute_metrics.py $SMOKE

echo "[phase=c_judge_validity]"
uv run python scripts/issue685_judge_validity.py $SMOKE

echo "[phase=d_figures]"
uv run python scripts/issue685_make_figures.py $SMOKE

if [ -z "$SMOKE" ]; then
  echo "[phase=upload]"
  uv run python scripts/issue685_upload.py
fi

echo "[phase=done]"
