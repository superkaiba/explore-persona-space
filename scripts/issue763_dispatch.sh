#!/usr/bin/env bash
# Issue #763 dispatch — the --workload-cmd driver (GCP / RunPod lane).
#
# UNIFIED end-to-end: this script runs the SAME python entrypoint chain in both
# real and --smoke modes; --smoke threads through to each phase (tiny slice:
# 1 behavior x 3 contexts x 5 probes, CPU, no API/HF). One launch covers ALL
# phases (build pools -> source-side baseline -> generate -> capture ->
# PV extract -> judge -> fit -> figures -> upload). No separate smoke/sweep
# code path (the PASS_UNIFIED smoke-architecture contract).
#
#   bash scripts/issue763_dispatch.sh                # full run (5 behaviors x 50 ctx)
#   bash scripts/issue763_dispatch.sh --smoke        # tiny offline verification (no pod/GPU/API)
#
# Phases:
#   build_pools         author + freeze + HF-upload the 5 eliciting pools (CPU/API)
#   source_side_baseline base-model propensity read (predicts low_dynamic_range; CPU/API)
#   generate            on-policy completions per (context x probe)            (GPU/vLLM)
#   capture             matched-probe teacher-forced v0(C,B) at all 28 layers  (GPU)
#   pv_extract          faithful persona-vector r_B (baseline arm)             (GPU/API)
#   judge               E0(C,B) via Sonnet (rubrics verbatim) + structural fmt (CPU/API)
#   fit                 GLM (primary) / ridge / PV LOCO + nulls + ceiling      (CPU)
#   figures             the §6 hero grid + exploratory dump                    (CPU)
#   upload              raw completions + v0/r_B analysis tensors -> HF        (CPU, real only)

set -euo pipefail

# REPO_ROOT defaults to the GCP workload root or the RunPod path; honored if
# pre-exported by the dispatch (#641: GCE startup exports it). The bash
# per-command `REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue763_dispatch.sh`
# form supersedes for that one command (belt-and-suspenders).
REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
if [ ! -d "$REPO_ROOT/scripts" ]; then
  # Fall back to the dir this script lives in (local VM smoke).
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

SMOKE=""
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE="--smoke" ;;
  esac
done

# Load credentials for the judge / HF (set -a && source .env && set +a per
# research-project-structure.md — never bare load_dotenv in a heredoc).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "[issue763.dispatch] REPO_ROOT=$REPO_ROOT SMOKE='${SMOKE}' commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# --smoke = 1 behavior x 3 contexts x 5 probes, fully offline (CPU, no API/HF).
# The real run = 5 behaviors x 50 contexts x ~60 probes on the GPU.
if [ -n "$SMOKE" ]; then
  SMOKE_MODEL="${EPM_SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  BEH="deception"

  echo "[phase=build_pools]"
  uv run python scripts/issue763_build_probe_pools.py --smoke --behaviors "$BEH"

  echo "[phase=source_side_baseline]"
  uv run python scripts/issue763_source_side_baseline.py --smoke --behaviors "$BEH" \
    --no-vllm --mock-judge --model-name "$SMOKE_MODEL"

  echo "[phase=generate]"
  uv run python scripts/issue763_generate_completions.py --smoke --behaviors "$BEH" \
    --n-contexts 3 --no-vllm --model-name "$SMOKE_MODEL"

  echo "[phase=capture]"
  uv run python scripts/issue763_capture_v0_matched.py --smoke --behaviors "$BEH" \
    --device cpu --model-name "$SMOKE_MODEL" --batch-size 4 --check-equivalence

  echo "[phase=pv_extract]"
  uv run python scripts/issue763_extract_pv_rb.py --smoke --behaviors "$BEH" \
    --mock --device cpu --model-name "$SMOKE_MODEL"

  echo "[phase=judge]"
  uv run python scripts/issue763_judge_e0.py --smoke --behaviors "$BEH" --mock-judge

  echo "[phase=fit]"
  uv run python scripts/issue763_fit_predictors.py --smoke --behaviors "$BEH"

  echo "[phase=figures]"
  uv run python scripts/issue763_plot.py --smoke --behaviors "$BEH"

  echo "[phase=done]"
  exit 0
fi

# ── REAL RUN ──────────────────────────────────────────────────────────────────
# Build pools is normally run OFF-pod before provision (CPU/API) so the frozen
# pools are HF-uploaded for the git-clone-only lane to snapshot_download. On the
# pod we stage them: if absent, snapshot_download the HF inputs mirror; if the
# builder must run here (API present), run it.
echo "[phase=build_pools]"
uv run python scripts/issue763_stage_pools.py || \
  uv run python scripts/issue763_build_probe_pools.py

echo "[phase=source_side_baseline]"
uv run python scripts/issue763_source_side_baseline.py

echo "[phase=generate]"
uv run python scripts/issue763_generate_completions.py

echo "[phase=capture]"
uv run python scripts/issue763_capture_v0_matched.py --device cuda

echo "[phase=pv_extract]"
uv run python scripts/issue763_extract_pv_rb.py --device cuda

# Upload raw completions + analysis tensors BEFORE the GPU pod is released
# (Upload Policy: raw completions + plan-referenced analysis tensors must land
# on HF before pod termination). The judge + fit are CPU-only and run off-pod.
echo "[phase=upload]"
uv run python scripts/issue763_upload.py

echo "[phase=judge]"
uv run python scripts/issue763_judge_e0.py

echo "[phase=fit]"
uv run python scripts/issue763_fit_predictors.py

echo "[phase=figures]"
uv run python scripts/issue763_plot.py

echo "[phase=done]"
