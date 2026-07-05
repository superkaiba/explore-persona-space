#!/usr/bin/env bash
# Issue #763 rubric-v2p1-disclosure-carveout GPU-phase driver (plan v10 §3d + v17 marker corrections).
# Chains the reviewed entrypoints verbatim; HF mirror is the lane return path (v2p1_outputs/).
set -euo pipefail
OUT=eval_results/issue_763/rubric-v2p1-disclosure-carveout
echo "[phase=fit_v2p1]"
EPM_FIT_DEVICE=cuda uv run python scripts/issue763_fit_predictors.py --behaviors deception \
  --e0-json $OUT/E0_deception_v2p1.json --out-dir $OUT \
  --binary-control-ref eval_results/issue_763/fit_by_behavior/deception.json --binary-control-tol 1e-3
echo "[phase=fit_ablate]"
EPM_FIT_DEVICE=cuda uv run python scripts/issue763_fit_predictors.py --behaviors deception \
  --e0-json $OUT/E0_deception_v2_ablate.json --out-dir $OUT/ablate \
  --binary-control-ref eval_results/issue_763/fit_by_behavior/deception.json --binary-control-tol 1e-3
echo "[phase=refresh_metadata]"
uv run python scripts/issue763_fit_predictors.py --refresh-metadata-only \
  --behaviors fact_expression format_style self_report persona_drift \
  --parent-fit-dir eval_results/issue_763/deception-rubric-reanchor/fit_by_behavior --out-dir $OUT
echo "[phase=verdict]"
EPM_FIT_DEVICE=cuda uv run python scripts/issue763_v2p1_verdict.py --out-dir $OUT
echo "[phase=figures]"
uv run python scripts/issue763_plot.py --results-json $OUT/matched_predictor_results.json \
  --compare-results eval_results/issue_763/deception-rubric-reanchor/matched_predictor_results.json \
  --base-e0 eval_results/issue_763/deception-rubric-reanchor/E0_deception_v2.json \
  --out-prefix v2p1 --tag-compare "v2 (re-anchored)" --tag-results "v2.1 (carve-out)"
echo "[phase=upload]"
uv run hf upload superkaiba1/explore-persona-space-data "$OUT" issue763_matched_v0/v2p1_outputs \
  --repo-type dataset --exclude "raw_completions/**" "smoke/**"
uv run hf upload superkaiba1/explore-persona-space-data figures/issue_763 issue763_matched_v0/v2p1_outputs/figures \
  --repo-type dataset --include "fig_763_v2p1*"
echo "[phase=done]"
