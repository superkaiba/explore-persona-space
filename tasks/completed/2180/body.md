---
title: Headline matched-partial needs a support-restricted companion when the matching
  covariate is degenerate on most of the sample
kind: infra
tags: []
created_at: '2026-08-07T16:18:06Z'
has_clean_result: false
origin_prompt: /issue 2163
workflow: v1
---
## Goal

Add a review lens requiring a **support-restricted companion read** whenever a headline correlational statistic's matching or stratification covariate is structurally zero — or heavily tied — on a large majority of the sample. In that regime "matched" does no work on the dominant block, and the headline can be carried entirely by the degenerate sub-population while every mechanical check passes.

## The gap

The project has strong machinery for selection-symmetric nulls (`.claude/rules/selection-symmetric-nulls.md`) and for measurement validity, and both worked as designed on the driving incident: the permutation band was exchangeable, selection rode inside every draw, the band recomputed exactly, and the positive controls passed. What no lens asks is whether the **matching covariate has support** on the sample the headline is computed over.

When the matching variable is identically 0 on most rows, its rank is one giant tie there, so the partial removes nothing and the stratified permutation has most of its mass in a single stratum. The statistic stays valid as a statistic. What breaks is the *narration*: the effect can live entirely in the block where matching is inert, and the prose attributes it to the mechanism the matched design was supposed to isolate.

## Driving incident (#2163, 2026-08-07)

Goal: which SAE features the context→answer map READS at the context vector `v_C`. Headline: activity-matched partial Spearman between 25 mechanical feature properties and `log U_j`, max |partial| 0.239 vs a 0.0092 selection-symmetric band — a 26× margin, matching variable `lasttoken_count`.

The partials ran over ~128-131k features. Only **13,289** ever fire at `v_C`; the other **117,783** never do, so `lasttoken_count` is identically 0 for ~90% of the sample. Splitting the headline by population (orchestrator recompute, independently reproducing the interpretation-critic's):

| population | n | `proj_var` | `scaffold_frac` |
|---|---|---|---|
| full | 131,072 | −0.243 | −0.210 |
| train-active at `v_C` | 13,289 | **−0.003** | **+0.015** |
| never-active | 117,783 | −0.263 | −0.239 |

The two named headline predictors carry ~zero signal among features that actually fire at `v_C`, and `scaffold_frac` flips sign. The promoted title — "per-unit gain is lowest on high-variance directions and template-heavy features" — read as a claim about reading at the context vector, was not supported; it described the map's ridge-coefficient geometry over decoder directions that never fire there. (Structure *does* survive within-active, with a different profile — activity −0.22, footprint_kurt −0.21, consistency +0.14 — so the finding was recoverable, just not as worded.)

Corroboration that was already sitting in the committed artifact and unread: on the frequency-weighted DV `A_W` (nonzero only on firing features) `proj_var`'s partial is **+0.038** — sign-flipped and outside its own band. The composition effect was visible in the same JSON.

**What makes this worth a lens:** it survived the planner, the full critic ensemble, the consistency-checker, the code-review ensemble, and the analyzer. Only a raw-artifact recompute by the interpretation-critic caught it — i.e. it is invisible to every check that reads the plan or the reported numbers rather than re-slicing the data.

## Proposed lens

Owner: `statistics-critic` (workflow v2) + `critic-lens-reference.md` Statistics & Measurement lens (v1), plus `interpretation-critic` since the driving catch happened at interpretation time.

Trigger: a plan or a promoted body carries a headline correlational / partial / matched statistic whose matching, stratification, or conditioning covariate is zero-or-tied on more than ~50% of the analysis sample.

Requirement: report the headline statistic **restricted to the covariate's support** alongside the full-pool value, and attribute the narration to whichever population carries it. A full-pool-only headline in this regime is a REVISE.

Mechanizable form worth considering: the analysis artifact already records `n_complete_case` and the matching column, so a check can compute the tied fraction of the matching covariate and require a `*_on_support` companion field whenever it exceeds the threshold. That would make the lens enforceable rather than advisory.

## Acceptance criteria

1. The lens exists in the statistics/measurement lens roster with its trigger and the support-restricted companion requirement.
2. `interpretation-critic` carries the matching-lens check so the interpretation-time catch is not luck next time.
3. The threshold and the "tied fraction" definition are written down (a covariate can be degenerate by ties without being zero).
4. Tests pin whatever mechanical form lands.
5. If a mechanical check is added, it fires on #2163's own committed `predictor_partials.json` as a regression fixture — that artifact is the known-positive case.

## Provenance

Surfaced by the `interpretation-critic` round 1 on #2163 as a prose workflow follow-up (it declined to file itself, per the subagents-never-file rule) and routed here by the orchestrator. Verified independently by orchestrator recompute over `read_ladder__W.npz` + `census.npz` + `inputs/fullwidth_covariates_v2.npz` before filing. Distinct from the sibling infra tasks filed from this session: #2171 (stub-grant token wiring), #2172 (parenthesized planned_wall_h cell), #2174 (sample-derived exact-identity premises), #2176 (hand-listed smoke-arch arm enumeration).
