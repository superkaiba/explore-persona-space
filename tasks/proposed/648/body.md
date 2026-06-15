---
title: Is centered cosine a better leakage predictor than raw cosine, or only the
  valid one? (head-to-head CV R² per bank)
kind: analysis
tags: []
created_at: '2026-06-15T22:40:29Z'
has_clean_result: false
parent_id: 536
origin_prompt: 'queue a centered-vs-raw predictive-skill comparison on the cached
  predictor panels (CV R²_raw vs CV R²_centered on the same cell): #536 settled which
  recipe is valid + that calls survive, but not which maximizes predictive skill;
  effect on skill goes both directions by bank (100-persona raw inflates 0.60->0.77;
  19-bank headline exists only centered -0.348 vs -0.037)'
---
## Goal

Determine whether mean-centering the centroid bank before computing cosine distance (the canonical recipe pinned by #536) yields a **better leakage predictor** than raw (un-centered) cosine — a head-to-head predictive-skill comparison (CV R²_raw vs CV R²_centered, and length-partialled Spearman ρ, on the same predictor cells), distinct from #536's robustness/validity audit.

#536 settled "which recipe is *valid*" (centered — raw is anisotropy-degenerate) and "do the published calls *survive* the swap" (yes). It did NOT ask "which recipe maximizes *predictive skill*." This task asks exactly that.

## Why it isn't already answered

#536 reported paired (raw, centered) rank correlations as a survival check, not predictive skill as the dependent variable — and the effect of centering on measured skill goes **both directions** depending on the bank:

- **100-persona bank:** raw *inflates* the headline — pooled ρ 0.60 (centered) → 0.77 (raw); raw overstates.
- **19-bank task:** the headline *exists only under centering* — ρ −0.348 (centered) vs −0.037 (raw); raw kills it.
- **cosine↔divergence alignment:** ρ 0.94 (raw) → 0.79 (centered) — part of the raw value is a shared-mean compression artifact.

So whether centering raises or lowers out-of-sample predictive skill is genuinely open and not monotone across banks.

## Formalization

- **Construct:** leakage-prediction skill of the cosine-distance predictor under {raw, centered} recipes.
- **Metric:** out-of-sample CV R² (leave-one-cond/source-out, matching each bank's published scheme) + length-partialled Spearman ρ of cosine distance against that bank's ΔG / leakage target, computed identically for raw (`centering='none'`) and centered (`centering='global_mean'`) on the SAME cells/banks.
- **DV:** the paired difference CV R²_centered − CV R²_raw per bank, with a paired bootstrap CI.

## Competing hypotheses (what counts as an answer)

- **H_centered-better:** centering raises out-of-sample CV R² (it strips shared-mean nuisance variance) → centered is both valid AND the stronger predictor everywhere.
- **H_bank-dependent:** the sign of the centered−raw skill difference depends on the bank (raw inflates some, centering rescues others) → no universal predictive-skill winner; centering is preferred on validity grounds, not skill.
- **H_raw-inflates:** raw cosine systematically inflates apparent skill via the anisotropy ridge → raw's higher numbers are artifact and centered is the honest predictor even where its CV R² reads lower.

An answer = a per-bank table of (CV R²_raw, CV R²_centered, paired Δ, CI) across the recoverable banks #536 used, plus a verdict on which hypothesis holds, tied back to #536's bank-dependent observation.

## Implementation (no new GPU; CPU on cached artifacts)

- Reuse #536's recompute driver `scripts/issue536_recompute_driver.py` join builders + the cached centroid banks it already consumed (`family_111bank` / `family_20bank` / `family_505` etc.). For each recoverable bank, compute the cosine-distance predictor under both centerings, regress against the same target with LOCO CV R² + length-partial Spearman, paired bootstrap CI.
- Restrict to banks whose centroids persist (the #536 join gate — raw recompute must reproduce the published number first). Matrix-only banks can give only the sensitivity-namespace approximate read and must be labeled as such, never pooled with exact rows.
- Same cached-residuals / centroid-bank footprint as #647; pairs naturally with it (both interrogate the geometry→leakage line).

## Caveats to carry into the result

- Centered cosine is only comparable **within a bank** (#536 pin) — never pool CV R² across banks into one number; report per-bank.
- Flag any saturated / degenerate cells.
- This compares predictive SKILL only; it does NOT relitigate validity — centering stays canonical regardless of the skill verdict.

## Provenance

- Surfaced in the #536/#589/#523/#522 review; user-requested ("queue that as a sub-question — a centered-vs-raw predictive-skill comparison on the cached predictor panels").
- Parent / context: #536 (metric-recipe audit), #502 / #511 (predictor sweeps), #474 (ΔG target matrix), #66 / #274 (centroid banks).
- Open question: `leak-predictor` / anchor #2 (do the published persona-distance / geometry leakage calls survive honest re-grading?).
