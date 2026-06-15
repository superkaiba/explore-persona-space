---
title: Is the geometry→behavior-strength relationship super-linear (convex), and does
  that shape recur across behaviors?
kind: experiment
tags: []
created_at: '2026-06-15T18:02:38Z'
has_clean_result: false
origin_prompt: In 623 we see kind of an exponential relationship between cosine similarity
  and sycophancy. I feel like we've seen this trend with a lot of the behaviors. Can
  you make and dispatch an issue in the background to look at all past results and
  explore this hypothesis?
goal: Across all past experiments that produced paired (persona-geometry scalar, behavior-strength
  scalar) data, characterize the FUNCTIONAL FORM of the geometry→behavior-strength
  relationship — testing on RAW (non-rank) values whether it is consistently convex/super-linear
  ('exponential-looking') rather than linear, and whether that shape recurs across
  behaviors (sycophancy, marker leakage, fact leakage, refusal, EM) after controlling
  for the fact-line sign flip, saturation/floor, high-leverage points, log-space artifacts,
  and the X vs (X−Y) caveat.
relates_to:
- leak-predictor
---
# Is the geometry→behavior-strength relationship super-linear (convex / "exponential-looking"), and does that shape recur across behaviors?

## Goal

Across all past experiments that produced paired (persona-geometry scalar, behavior-strength scalar) data, characterize the FUNCTIONAL FORM of the geometry→behavior-strength relationship — testing on RAW (non-rank) values whether it is consistently convex/super-linear ('exponential-looking') rather than linear, and whether that shape recurs across behaviors (sycophancy, marker leakage, fact leakage, refusal, EM) after controlling for the fact-line sign flip, saturation/floor, high-leverage points, log-space artifacts, and the X vs (X−Y) caveat.

## Hypothesis

Across behaviors, a persona's (or unit's) geometric proximity to a behavior — cosine similarity to the behavior direction, or cosine/JS distance to the source/teaching persona — relates to that unit's behavior strength in a **convex, accelerating** way rather than linearly: behavior strength stays near a floor across most of the proximity range, then rises steeply once proximity passes some point. The #623 sycophancy scatter (per-persona cosine to the sycophancy direction vs base sycophancy rate) reads this way by eye. The claim under test is that this convex/super-linear shape is a recurring property of geometry→behavior relationships, not a one-off of sycophancy.

## Why this is worth a task

The standing leak-predictor line (q:leak-predictor) has so far asked a coarser question — does geometry predict behavior *at all*, monotonically (Spearman rho / Pearson r)? — and answered "inconsistently across behaviors": cosine/JS to source predicts marker leakage (#207, #311, cosine gradient r≈0.54-0.83), #411 found a partial sycophancy cosine gradient, #623 found cosine→base-sycophancy rho=0.73, while the fact line (#444) predicts with the WRONG sign (the reference frame, not the probe slice, flips it) and #532/#541 found a unit's behavioral base prior keeps out-predicting geometric cosine. None of this has examined the **functional form**. Rank correlations (Spearman) are blind to shape by construction — they cannot distinguish linear from exponential. A convex shape, if it recurs, is a sharper and more useful statement than "monotonic": it implies a proximity threshold below which behavior is geometrically uncoupled, which bears directly on the pre-training-audit application (q:app5) and on whether geometry is a usable safety predictor.

## Scope: re-analyze existing paired data first

This is primarily a meta-analysis / re-analysis of data already produced. The first cut should require little or no new GPU. Inventory every past task that produced paired **(geometry scalar, behavior-strength scalar)** points across a panel of personas/units, pull the RAW (non-rank) values, and fit and compare functional forms. Candidate source tasks (planner to confirm what raw data actually survives on disk / HF):

- **#623** — per-persona cosine(persona vector, sycophancy direction) vs judged base sycophancy rate, n=35, raw scatter in `eval_results/issue_623/` (`cosine_matrix.json`, `syc_i.json`). The seed observation.
- **#411** — sycophancy cosine gradient across source personas on held-out wrong claims.
- **#207 / #311** — marker leakage vs cosine/JS distance to source persona (the original distance→leakage gradient; cosine r≈0.54-0.83).
- **#383** — selectivity-recipe gradient (carry the X vs (X−Y) spurious-correlation caveat).
- **#404 / #458** — persona-distance predictors (cosine / JS-divergence operationalizations).
- **#444 / #500** — fact-leakage, where teacher-referenced distance predicts with the wrong sign; the bystander's base prior predicts positively.
- **#532 / #541** — base behavioral prior vs geometric cosine for leakage.
- Any EM / trait / refusal task with a per-unit geometry scalar paired to a per-unit behavior-strength scalar (#390 refusal, EM line).

## What would count as an answer

Per behavior with paired raw data:

1. Fit and compare candidate functional forms on the raw (geometry x → behavior-strength y) scatter: **linear vs convex** (exponential `y = a·exp(b·x)`, power-law, quadratic, or a monotone spline with a curvature/convexity test). Compare by held-out fit (LOO / k-fold predictive R²) and AIC/BIC, not in-sample R² alone. Report a direct **convexity test** (e.g. sign of the fitted curvature term with bootstrap CI, or the cubic/quadratic vs linear nested test).
2. Report whether convex beats linear, with effect size and CI, and whether the result is robust to the high-leverage points (#623-style leave-one-out).

Cross-behavior synthesis:

3. Does the convex/super-linear shape recur, and with a consistent sign? Explicitly fold in the fact-line sign flip (#444) — if the reference-frame hypothesis is right, the shape claim should be stated relative to the correctly-signed proximity, not raw cosine-to-an-arbitrary-teacher.

## Competing hypotheses

- **H1 (convex/super-linear recurs):** behavior strength is a convex function of geometric proximity across behaviors; small proximity buys ~floor behavior, then it accelerates.
- **H2 (monotone but linear / artifactual exponential):** the #623 "exponential look" is an artifact — a few high-leverage personas, the rank metric obscuring a roughly linear raw relationship, or axis-scaling — and the raw relationship is no more than linear once leverage is controlled.
- **H0 (no consistent shape):** geometry→behavior shape is behavior-specific (consistent with the current q:leak-predictor "inconsistent across behaviors" belief); there is no portable functional form.

## Measurement-validity notes (for the planner)

- **Work on RAW values, not ranks.** Spearman rho (the #623 headline) measures monotonicity, not shape. Testing "exponential" requires the raw scatter and a functional-form / curvature test. This is the central measurement-validity point of the whole task.
- **Comparability across behaviors.** The geometry scalar (cosine-to-direction vs cosine/JS-to-source) and the behavior-strength scalar (base rate, leakage rate, marker log-prob, judged trait rate) differ across the source tasks. State per behavior what each axis is and whether shapes are being compared on commensurable scales (e.g. standardize x within behavior; be explicit that a log-prob DV is already in log space, which mechanically manufactures convexity when mapped back to probability — a key confound to neutralize).
- **Saturation / floor.** A behavior strength saturated at a ceiling or pinned at a floor across most units manufactures apparent curvature; require spread on both axes (as #623 documented) before fitting a shape.
- **X vs (X−Y) caveat** (#383): do not regress a difference against one of its own components.
- Where existing data is too sparse per behavior to fit a shape, the deliverable is a clear statement of which behaviors have enough points and which need new generation — propose the new runs as follow-ups rather than silently under-powering a fit.

## Provenance

Created from a chat request to investigate whether the #623 cosine-similarity → sycophancy relationship (which looks roughly exponential) generalizes across behaviors. Routed as a NEW direction (a question about the functional form of geometry→behavior, not answerable by rewriting any single existing issue's takeaways).
