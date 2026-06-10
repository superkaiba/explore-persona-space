---
title: 'Per-cohort geometry-on-residual: does activation distance rank cells within
  fixed base-prior bands?'
kind: experiment
tags: []
created_at: '2026-06-09T20:31:59Z'
has_clean_result: false
parent_id: 532
goal: 'Test whether the geometric predictors (cosine, Gaussian-KL@L22) carry rank-information
  about on-policy marker emission inside narrow base-prior bands of the 416-cell ep1
  panel from #532, by computing Spearman ρ between each geometric predictor and the
  leakage residual after the base prior is regressed out per-bystander.'
relates_to:
- leak-predictor
- spec-sysprompt-vs-drift
---
## Goal

Test whether the geometric predictors (cosine, Gaussian-KL@L22) carry rank-information about on-policy marker emission inside narrow base-prior bands of the 416-cell ep1 panel from #532, by computing Spearman ρ between each geometric predictor and the leakage residual after the base prior is regressed out per-bystander.

## Motivation

#532's headline rests on a 6-regression CV hierarchy: ΔR²-prior-beyond-flag = +0.172, ΔR²-geometry-beyond-flag-plus-prior = +0.014. This summary is *aggregated*: it tells us geometry adds 1.4 percentage points of variance ON TOP OF the binary instructed/ordinary indicator + base prior, but does not tell us whether that 1.4 ppt is concentrated in one cohort (e.g., the cross-class ordinary subset) or spread thin across both. Without this per-cohort residual ρ the body's "geometry isn't useless, it's redundant" framing is provisional.

A per-cohort geometry-on-residual regression resolves the question directly: regress leakage on the per-bystander base prior, then correlate each geometric predictor against the residual within (a) the cross-class ordinary subset, (b) the instructed strip. If the geometry-on-residual ρ is near-null in both cohorts, the headline strengthens to "geometry tracks the cohort flag and nothing else." If geometry recovers a non-trivial ρ on the cross-class ordinary subset (replicating #502's −0.79 finding in residual form), the +0.014 ΔR² is a per-cohort effect, not a uniform one.

## Hypotheses

- **H1:** Geometric predictor ρ on the leakage residual is ≥ +0.15 (absolute) on the cross-class ordinary subset (where the base prior is ~0).
- **H2:** Geometric predictor ρ on the leakage residual is < |0.10| on the instructed strip.

## Proposed design (planner owns the final spec)

- Reuse the 416 per-cell JSONs at `eval_results/issue_532/per_cell/loc_ep1/` (unchanged).
- Reuse `predictors.json` and `phase0_base_prior.json` from #532.
- Add a new analysis step (`phase3_residual_per_cohort` or equivalent) that, per bystander, regresses leakage on the base prior, computes the residual, then computes the Spearman ρ between each of (cosine, JS-v1, GKL@L22) and the residual within the cross-class ordinary subset + instructed strip.
- Report ρ with permutation-test p-values (1000 reps) and 95% bootstrap CIs.

## What we learn

A definitive per-cohort answer to "does activation distance ever earn its keep beyond the cohort indicator + base prior?". Strengthens or weakens the geometric program independently of the cohort confound and finishes the §6.2 hierarchy that #532's headline rests on.

## Reuse / cost

- **Cost:** ~0 GPU-hours (analysis-only re-run over committed JSONs).
- **Reuse:** parent #532's full per-cell panel + predictor matrices + base-prior payload; no training, no eval generation, no pod.
