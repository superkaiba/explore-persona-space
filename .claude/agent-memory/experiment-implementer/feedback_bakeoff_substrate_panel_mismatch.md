---
name: Bakeoff regression hardcodes #474 G_logprob matrix path
description: issue493_extraction_metric_bakeoff.py _load_G reads eval_results/issue_474/cross_eval/{arm}_ep{ep}/G_logprob_matrix.json with #474 cond_ids (A1-A5/B1-B11); on new panels (R1..R24 for #518) regression crashes with KeyError. Use --phase metrics (split extract + metrics) to skip the broken regression entirely — the metrics-phase output is sufficient for issue509_scoring.py which never reads bakeoff regression.
type: feedback
---

When cloning the #493/#509 residual-stream bake-off to a new conditions panel (e.g. #518 R1..R24 refusal / E1..E24 em panels), the regression phase will crash mid-run with `KeyError: 'R1'` because `_load_G(arm, ep)` at line 368 is hardcoded to read `eval_results/issue_474/cross_eval/{arm}_ep{ep}/G_logprob_matrix.json` — which contains the #474 A1-A5/B1-B11 16-cond panel. The regression phase produces an intra-arm "winning predictor" leaderboard that is OPTIONAL for downstream consumers.

**Why:** The headline cross-arm comparison flows through `issue509_scoring.py --metrics-dir` → `issue518_cross_behavior_aggregator.py` which reads ONLY the per-(point, layer, metric, variant) metric JSONs from `<bakeoff-root>/metrics/`, never the regression's `bakeoff_grid.json` or `regression/loc_ep1.json`. Scoring's `per_coarse_rho_fe` is computed internally from the substrate's `predictor_comparison.json`, not from the bake-off regression.

**How to apply:** In the launcher, pin the bake-off invocation to two explicit phases instead of `--phase all`:
```
uv run python scripts/issue493_extraction_metric_bakeoff.py ... --phase extract
uv run python scripts/issue493_extraction_metric_bakeoff.py ... --phase metrics
```
This stops at line 5380 (`if args.phase == "metrics": return 0`), BEFORE the regression. Both phases are idempotent on resume — `run_extraction_batched` skips per-cond partitions whose canonical exists, `run_metrics` skips per-cell JSONs that exist. `meta.json` is written at every entry (line 5051) so the aggregator's `_assert_model_id_consistency` is satisfied.

Closed task #518 round-14 (2026-06-09). Crash signature: 11h of upstream work + KeyError at run_regression line 3421 = `np.array([G[a][b]["delta_g"] for a, b in pairs_full16])` where `pairs_full16` ranges over R1..R24 but G's keys are A1-A5/B1-B11.

Do NOT try to add a `--g-matrix-path` flag + write a #518→G adapter to feed the regression — that produces a leaderboard the #518 Goal doesn't consume and burns ~1-2h of additional compute. The cleanest fix is to skip the regression entirely; the substrate + scoring + aggregator chain produces the headline answer directly.
