---
title: Does the base-prior-vs-geometry level/change split hold for realistic behaviors
  (sycophancy primary)?
kind: experiment
tags: []
created_at: '2026-06-15T23:40:47Z'
has_clean_result: false
parent_id: 532
origin_prompt: Can we run a followup to check effect of base prior on change in leakage
  for the more realistic behaviors?
---
# Does the base-prior-vs-geometry level/change split hold for realistic behaviors?

## Goal

Test whether the #532 **level-vs-change predictor decomposition** — the base-model
prior predicts the *absolute level* of leakage while activation geometry predicts
the *training-induced change* (trained − base) — holds for a realistic judged
behavior (sycophancy primary; refusal as a stretch), rather than being specific to
the programmatic marker.

Per (source × bystander) cell, decompose leakage into:
- **LEVEL** = absolute trained expression of the behavior at the bystander, and
- **CHANGE** = the trained − base shift at the bystander,

and correlate each against (a) the **bystander's base prior** on the behavior
(base-model expression rate / probability under that bystander context) and
(b) **source↔bystander activation geometry** (cosine @ the #509 early-layer band,
Gaussian-KL), with the prior partialled out — the #532 six-regression
CV-R² hierarchy, reported separately for the LEVEL DV and the CHANGE DV.

**Competing hypotheses:**
- **H1 — marker pattern replicates:** prior predicts LEVEL, geometry predicts
  CHANGE; prior ≈ 0 on CHANGE (the clean two-component rule from #532/#531).
- **H2 — behavior-specific:** for a realistic behavior the base prior *also*
  predicts the CHANGE (the implant rides the model's pre-existing propensity), so
  the marker's clean split does not generalize.
- **H0:** neither predictor survives on the CHANGE DV (floor/noise — plausible
  given sycophancy rate-space floors).

What counts as an answer: the per-DV incremental-validity ladder (ΔCV-R² of prior
beyond a cohort/indicator flag, geometry beyond flag+prior) on the LEVEL DV and the
CHANGE DV, plus the marginal Spearman of each predictor vs each DV — directly
comparable to #532's table (prior +0.176 ΔR² on level / −0.0005 on change;
geometry +0.021 on level / +0.215 on change for the marker).

## Motivation

The base-prior-beats-geometry result is now cross-behavior for the *level* of
leakage — marker (#532, HIGH) and facts (#500/#541, MODERATE) — and #532's
corrected-slot follow-up sharpened it into a two-component rule: **prior forecasts
where the behavior already sits (level); activation distance forecasts what training
moved (change).** That rule has only ever been tested on the marker. The whole
predictor program's safety-relevant claim ("where will an implanted behavior surface
after training") is about the CHANGE, and we have never checked whether the prior is
the null on the change for a *realistic* behavior the way it is for the marker.

The sycophancy line is adjacent but does not answer this:
- **#509** (7B): early-layer geometry predicts sycophancy leakage *beyond* the
  bystander base rate (geometry unique R² 0.191 vs base-rate 0.034) — but treats
  leakage (the lift) as a single DV; no absolute-level vs change split.
- **#507** (72B): the content-free bystander base rate *beats* all geometry on
  sycophancy leakage — the opposite ranking, and again a single-DV race.
- **#544** (on_hold): whether #509's early-layer geometry recovers sycophancy
  leakage at 72B — still a single-DV predictor race, not the level/change split.
- **#612/#627**: graded-cosine sycophancy panels with base-rate controls + matched
  install — they carry base prior + trained rate + cosine, but read install
  strength / selectivity, not the decomposition.

So #507/#509 already disagree on "prior vs geometry for sycophancy leakage" by
scale, which is *exactly* the symptom the level-vs-change split resolves: if leakage
(lift) mixes a prior-driven level component and a geometry-driven change component
in scale-dependent proportions, racing predictors on the undifferentiated lift will
flip rankings. Decomposing it is the principled fix.

## Feasibility / reuse (likely low-to-zero GPU)

Strong artifact reuse — the planner should check whether the decomposition is a pure
re-analysis before provisioning anything:
- **#612 / #627** — graded-cosine persona panels with base agreement rate (prior),
  trained agreement rate (→ level and change), and layer-20 cosine already on disk.
- **#507 / #509** — frozen 7B and 72B sycophancy leakage matrices + base rates +
  the early-layer geometry cells.
- **#391** — the forced-choice `P(user's-side)` sycophancy probe: a graded,
  log-prob-readable DV if the judge-scored agreement *rate* floors and kills the
  change-DV resolution (the rate-space analog of the marker emission floor).

If a graded log-prob-readable DV is needed and not already present, that is the only
plausible GPU cost; otherwise this is a CPU re-analysis in the #532/#539 mold.

## Measurement-validity notes (carry into the plan)

- The CHANGE DV must be **trained − base at an on-policy slot** (judge-scored
  on-policy rate, or the #391 forced-choice probability shift) — never a
  teacher-forced read on a canned completion (#432→#456), and never absolute
  expression read as "leakage" (absolute = the prior, the whole #532 point).
- Sycophancy has no single canonical completion token, so the "prior" is the
  judge-scored base agreement rate (or the #391 forced-choice probability), not a
  log P(string) — see the completion-scoring constraints (surface-form competition,
  PMI/contrastive scoring) established in the 2026-06-15 deep-research pass.
- Watch both saturation zones (rate floors for sycophancy; the #480 log-prob
  inversion at heavily-trained checkpoints) and read the level/change split at a
  non-saturated install (matched-install checkpoints from #627 are the natural anchor).

## Relationship / dedup

Parent: #532 (the decomposition originates there). Siblings the planner must NOT
duplicate: #507, #509, #544 (single-DV geometry-vs-base-rate races), #612/#627
(install strength). The novel object here is the **per-DV level-vs-change
decomposition** on a realistic behavior — neither sibling computes it.

## Provenance

Originating user request (verbatim): "Can we run a followup to check effect of base
prior on change in leakage for the more realistic behaviors?" — chat session
2026-06-15, following a deep read of #532/#539/#540 and the establishment that the
base-prior-vs-geometry level/change split is, to date, marker-and-fact-only and
untested on judged/realistic behaviors.
