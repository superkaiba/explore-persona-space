#!/usr/bin/env bash
# issue #666 Phase 4 — end-to-end production dispatcher (plan §10 pipeline DAG).
#
# Runs the full pipeline:
#   (i)   corpus_extract.py   — broad-corpus Σ_c extraction (the ONE GPU step;
#                                ≥2-5k FineWeb contexts × Qwen-2.5-7B layer 14;
#                                uploads sigma_c_inv.pt to HF data repo)
#   (ii)  predictor.py        — full 50-cell sweep + cross-behavior grid +
#                                headline JSON + rb_source_sensitivity (CPU on pod)
#   (iii) lobo_loco.py        — LOBO + LOCO context-id folds (CPU)
#   (iv)  designed_null.py    — install-leak control on the 2 designed-null
#                                cells, broad-Σc parity-gated (CPU)
#   (v)   noise_floor.py      — probe-split test-retest MC (CPU)
#   (vi)  clustered_ci.py     — family-clustered + naive bootstrap CIs (CPU)
#   (vii) figures.py          — per-behavior hero figure + scatters (CPU)
#
# Exits 0 on full pipeline success, non-zero on first phase failure. Each phase
# emits its own [phase=<name>] sentinel line + writes its artifact JSON to
# eval_results/issue_666/. Outputs:
#   - data/issue_666/sigma_c_inv.pt          (broad-corpus Σc⁻¹; uploaded to HF)
#   - data/issue_666/sigma_c_corpus_vectors.pt (the broad-corpus context vectors)
#   - eval_results/issue_666/predictor/*_predictor_cells.json (50 per-cell)
#   - eval_results/issue_666/headline/predictor_headline.json (the §6.5 PRIMARY)
#   - eval_results/issue_666/headline/rb_source_sensitivity.json
#   - eval_results/issue_666/headline/designed_null_Lhat_rho.json
#   - eval_results/issue_666/lobo_loco/lobo_loco.json
#   - eval_results/issue_666/noise_floor/noise_floor.json
#   - eval_results/issue_666/clustered_ci.json
#   - figures/issue_666/hero_predictor_rho_by_behavior.{png,pdf,meta.json}
#   - figures/issue_666/scatter_<cell>.{pdf,meta.json}      (50 per-cell scatters)
#
# Compute: 1 GPU step (~1-2 GPU-h on a single A100/H100/L4) + ~10-30 min CPU.
# Approved plan budget: 4 GPU-h total (well-bounded).

set -euo pipefail

# Resolve REPO_ROOT against the worktree this script lives in (pod-side path
# is /workspace/eps-issue-<N> or similar; VM-side is the repo root). Both
# resolve correctly because the script lives at scripts/ under whatever the
# repo / pod-side root is.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

PHASE_LOG="${PHASE_LOG:-/tmp/issue666_dispatch.log}"
SIGMA_INV_PATH="data/issue_666/sigma_c_inv.pt"

echo "[dispatch] issue 666 Phase 4 — full pipeline launch at $(date -u +%FT%TZ)" | tee -a "$PHASE_LOG"
echo "[dispatch] REPO_ROOT=$REPO_ROOT" | tee -a "$PHASE_LOG"
echo "[dispatch] PHASE_LOG=$PHASE_LOG" | tee -a "$PHASE_LOG"

run_phase () {
  local name="$1"; shift
  echo "" | tee -a "$PHASE_LOG"
  echo "==================================================================" | tee -a "$PHASE_LOG"
  echo "[dispatch] PHASE: $name @ $(date -u +%FT%TZ)" | tee -a "$PHASE_LOG"
  echo "[dispatch] CMD: $*" | tee -a "$PHASE_LOG"
  echo "==================================================================" | tee -a "$PHASE_LOG"
  "$@" 2>&1 | tee -a "$PHASE_LOG"
}

# (i) Σ_c broad-corpus extraction — the ONE GPU step. Uploads sigma_c_inv.pt
# to HF data repo at issue666 prefix (per the script's internal upload path).
run_phase "corpus_extract" uv run python scripts/issue666_corpus_extract.py

# (ii) Full predictor sweep — production headline driver.
#      Threads the broad-corpus Σc⁻¹ + #658 r_b.pt diffmeans + per-cell r_plus.
#      Writes per-cell JSONs + predictor_headline.json + rb_source_sensitivity.json.
run_phase "predictor" uv run python scripts/issue666_predictor.py \
  --sigma-inv "$SIGMA_INV_PATH" \
  --r-b-source mixed \
  --rb-source-sensitivity

# (iii) LOBO + LOCO context-id folds.
run_phase "lobo_loco" uv run python scripts/issue666_lobo_loco.py

# (iv) Designed-null install-leak control arm (reads predictor's broad-Σc per-cell
#       JSONs by long name; fails loud on non-broad Σc parity violation).
run_phase "designed_null" uv run python scripts/issue666_designed_null.py

# (v) Noise floor — probe-split test-retest MC (200 × 3 = 600 resamples per the plan).
run_phase "noise_floor" uv run python scripts/issue666_noise_floor.py

# (vi) Family-clustered + naive bootstrap CIs (n_boot=2000 per plan §6).
run_phase "clustered_ci" uv run python scripts/issue666_clustered_ci.py

# (vii) Figures — hero + per-cell scatters via paper_plots (SHA-pinned PNG+PDF).
run_phase "figures" uv run python scripts/issue666_figures.py

echo "" | tee -a "$PHASE_LOG"
echo "[dispatch] [phase=done] full pipeline OK @ $(date -u +%FT%TZ)" | tee -a "$PHASE_LOG"
echo "[dispatch] HEADLINE: $(cat eval_results/issue_666/headline/predictor_headline.json | head -c 300)" | tee -a "$PHASE_LOG"
