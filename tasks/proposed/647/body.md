---
title: Do the activation-geometry predictors of marker-leakage transfer also collapse
  on the nonstylized subpanel, or is within-nonstylized signal real?
kind: analysis
tags: []
created_at: '2026-06-15T21:34:08Z'
has_clean_result: false
parent_id: 522
origin_prompt: 'look if it''s already filed and otherwise queue it in background (re:
  #522 follow-up — run the activation predictors on the same nonstylized subpanel
  to test whether the whole distance->leakage line is just the stylized split)'
---
## Goal

Determine whether the activation-geometry distance predictors of marker-leakage transfer (gauss_kl / mmd / wass2 / cosine at last-prompt-token, layer 22, cloud-aware ridge) carry **within-nonstylized** signal, or whether — like the full-response JS predictor (#522) — their apparent skill on the full 16-persona panel is driven almost entirely by the stylized-vs-nonstylized split.

This is the single cheap diagnostic that decides whether the whole distance→leakage line (#502 / #511 / #522 / #536) measures a continuous distance gradient among normal personas, or just detects three theatrical outliers (pirate captain / stand-up comedian / villainous mastermind) sitting far from everything else.

## Formalization

- **Construct:** predictor-vs-leakage skill of each activation distance metric, on the 156 ordered pairs of the 13 **nonstylized** personas only (the full 16-persona panel minus A3/A4/A5).
- **Metric:** leave-one-cond-out (LOCO) CV R² and length-partialled Spearman ρ of each metric against the #474 ΔG target (loc-arm, epoch 1 headline; epochs 2/3/5 as robustness), exactly the rig in `scripts/issue511_probe_count_sweep.py` / `issue493_extraction_metric_bakeoff.py`, with panel-row bootstrap CIs (n_boot=2000, seed=42).
- **Comparison anchors:** full-panel CV R² (gauss_kl ≈0.62, mmd ≈0.58, wass2 ≈0.57, cosine ≈0.56 at L22/ep1, from #522/#511) and #522's JS-on-nonstylized result (CV R² ≈ −0.02, CI straddles zero).

## Competing hypotheses (what counts as an answer)

- **H_stylized-only:** the activation predictors ALSO collapse on the nonstylized subpanel (CV R² → ~0, CIs straddle zero, like JS). Read: all distance predictors of marker-leakage transfer are basically detecting the stylized split; the geometry-leakage line generalizes poorly within normal personas. This would substantially weaken the headline of #502/#511 and feed open-question #2 ("do the published geometry-leakage calls survive honest re-grading?").
- **H_geometry-real:** the activation predictors keep meaningful CV R² on the nonstylized subpanel where JS does not. Read: activation geometry captures the within-nonstylized signal that output-distribution JS misses — the distance→leakage line is more than the stylized split, and activation geometry is the right substrate.

## Implementation (no new GPU; CPU on cached residuals)

- Extend `scripts/issue511_probe_count_sweep.py` to score on the nonstylized subpanel — call `bakeoff._pairs(cond_ids, nonstylized_only=True)` (the 156-pair restriction #522 used), reusing the already-cached residual streams from the #511/#522 Phase-1 sweep. No new training, no new generation.
- Run all four metrics × the L19–L24 band (headline L22) × loc-arm epochs {1,2,3,5}, mirroring #522's full-panel sweep so the full-vs-nonstylized contrast is matched cell-for-cell.
- Report each metric's full-panel vs nonstylized CV R² side by side, with the JS-on-nonstylized bar from #522 overlaid for reference.

## Known caveat (carry into the result)

LOCO on 13 highly-correlated nonstylized personas is itself a stress test with little held-out signal, so a negative CV R² is ambiguous between "the predictor genuinely doesn't generalize within nonstylized" and "the panel is too small for LOCO to resolve anything." If the activation predictors collapse, the clean disambiguating follow-up is to expand the nonstylized panel with a richer / more diverse set of everyday personas (#522 follow-up #2) before concluding there is no within-nonstylized gradient.

## Provenance

- Flagged as the natural next diagnostic in #522 ("Follow-ups worth queueing" #1) and surfaced in the #536/#589/#523/#522 review.
- Parent / context: #522 (full-response JS predictor), #511 (activation-predictor sweep), #502 (16-persona panel + headline cell), #536 (metric-recipe audit), #474 (ΔG target matrix), #406 (persona configs).
- Open question: `leak-predictor` / anchor #2 (do the published persona-distance / geometry leakage calls survive honest re-grading?).
