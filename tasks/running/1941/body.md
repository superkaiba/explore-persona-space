---
title: 'functional_role axis: kappa 0.318 reproduces at both scales — the only axis
  with no contrastive evidence'
kind: infra
tags:
- sae-instrument
created_at: '2026-07-31T16:35:50Z'
has_clean_result: false
origin_prompt: '#1773 full-dict run: functional_role kappa 0.318 (vs 0.63-0.71 for
  the other four) AND 2.39% drops (vs 0.51-1.01%), paired and stable across all 32
  groups; reproduces 0.310->0.318 across an 8x scale change while speaker_property
  improved 0.512->0.629'
workflow: v1
---
## Overview / Motivation

#1773's `functional_role` axis is reproducibly the worst of its five judged axes on BOTH quality measures, at BOTH dictionary scales. It should be diagnosed or retired before it is joined against in any downstream analysis.

## Goal

Establish why `functional_role` inter-draw agreement is roughly half the other four axes', and either fix its rubric/evidence or mark the axis unusable.

## The evidence

Inter-draw Fleiss kappa on the full 131,072-feature dictionary run (#1773, 2026-07-31, 3,212,800 calls):

| axis | kappa | content-drop rate |
|---|---|---|
| content_type | 0.708 | 0.45% |
| abstraction | 0.683 | ~0.7% |
| interpretable | 0.633 | ~0.9% |
| speaker_property | 0.629 | ~1.0% |
| **functional_role** | **0.318** | **2.39%** |

Two things make this a rubric-defect signature rather than sampling noise:

1. **Both defects are paired and co-located.** The worst-agreement axis is also the one with 3-5x the drop rate of its siblings, and the pairing was stable from group 1 through group 32 of 32 — no drift.
2. **It reproduces across an 8x scale change while a sibling improves.** #1773's original restricted (16,384-feature) run measured `functional_role` at kappa 0.310; the full dictionary gives 0.318. Over the same scale change `speaker_property` IMPROVED 0.512 -> 0.629. So four axes are stable-or-better and one is reproducibly bad at both scales. Scale is not the explanation.

## Leading hypothesis (testable, and cheap to test first)

**`functional_role` is the only axis that sees NO contrastive evidence.** From `AXIS_SEES` (`scripts/issue1773_common.py:175-181`):

| axis | evidence blocks | has a contrast? |
|---|---|---|
| abstraction | EX_POS, EX_NEG, DESC | yes (negatives) |
| interpretable | EX_POS, EX_NEG, DESC | yes (negatives) |
| speaker_property | EX_POS_DIVERSE, EX_NEG, NEAR, DESC | yes (negatives + neighbours) |
| content_type | EX_POS, NEAR, DESC | yes (neighbours) |
| **functional_role** | **EX_POS, OUT, DESC** | **NO — neither negatives nor neighbours** |

Every axis that clears kappa 0.6 sees either non-activating examples or nearest-neighbour features. `functional_role` sees activating examples, the logit footprint (`OUT` — it is the only consumer of that block), and the description. Judging whether a feature is `input_side` / `output_promoting` / `mixed` is an inherently RELATIONAL question, and it is the one axis given nothing to relate against. That predicts exactly the observed symptom: high draw-to-draw variance because there is no evidence that discriminates the options.

Cheap first test: re-run `functional_role` on a small stratified feature sample with `EX_NEG` and/or `NEAR` added, and see whether kappa moves toward the other axes'. A few hundred features at 5 draws is on the order of single-digit dollars at the realized #1773 basis (997.4 in / 163.8 out per axis call, Batch API).

Secondary hypotheses if that fails: the `mixed` label is a dumping ground absorbing genuine uncertainty; or the three-way distinction is ill-posed for features whose decoder input/output ratio sits mid-range (the `functional_role_side_ratio` mechanical validator in `eval_results/issue_1773/validation/mechanical_validators.json` is the non-judge reference to correlate against).

## Constraints / scope

- This does NOT reopen #1773's verdict. All five axes already read SEARCH-INDEX-ONLY on the neighbour-discrimination bar (0.322 vs 0.50, confirmed on a near-uncensored sample by the 2026-07-30 re-judge). `functional_role` is additionally unreliable at the draw level, which is a separate and narrower defect.
- Outcome may legitimately be RETIREMENT rather than repair: if a contrastive-evidence variant does not lift kappa, the honest action is to mark the axis unusable in `feature_table_v1.jsonl` consumers rather than ship a labelled column nobody should join against.
- The full-dictionary labels are already produced. This task is about whether the `functional_role` column in them is usable, not about re-running the other four axes.

## Provenance

Observed and reported by the #1773 full-dictionary run's implementer (epm:progress v43, 2026-07-31). The cross-scale comparison against the 16k run and the contrastive-evidence hypothesis were added by the orchestrator; the `AXIS_SEES` table above was read from source at compose time.
