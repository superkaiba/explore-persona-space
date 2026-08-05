---
title: 'batch drain loses bare-numeric judge verdicts (#1434 parity not ported): 64%
  vs 2.8% drops'
kind: infra
tags: []
created_at: '2026-08-05T19:56:34Z'
has_clean_result: false
origin_prompt: 'Found on #1739 item-A pilot judge: eval/batch_judge.py:390 drains
  with bare parse_judge_json while eval/judge_dispatch.py sync paths apply _normalize_scalar_score
  first. Same rubric/model/max_tokens: tom-gibbs content drops 64.17% batch vs 2.80%
  sync; a 4096-token batch re-judge still 63.70%, so budget is ruled out and route
  is the single variable. Corpus-dependent (2.7-64%), so it biases cross-rung comparisons
  differentially. Affects the trait_dv_1024 pool and any fleet batch wave on a just-the-number
  rubric.'
workflow: v1
---
## Overview / Motivation

Filed from task #1739 (evil-ood-spread-round), 2026-08-05. FLEET-WIDE, LIVE ON MAIN.

The Batch-API drain path in `src/explore_persona_space/eval/batch_judge.py` silently discards BARE-NUMERIC judge verdicts as `parse_error` drops, while every sync drain path in `eval/judge_dispatch.py` normalizes them first. The #1434 "dispatch-path parity" fix was applied to the sync paths and never ported to the batch drain — so the same rubric, model, and `max_tokens` produce wildly different drop rates depending only on the ROUTE, and the affected rubric family is the one the project uses most for graded trait scoring.

## Goal

Port `_normalize_scalar_score` (or its equivalent) into the batch drain so bare-numeric verdicts are recovered identically on both routes, and add a parity pin so the two paths cannot diverge again.

## The defect, verified in source on main

`src/explore_persona_space/eval/batch_judge.py` (~line 390), batch drain:

    parsed = parse_judge_json(text)
    score = parsed if parsed is not None else _legacy_error_dict("parse_error")

No scalar normalization. The sync drains in `eval/judge_dispatch.py` apply `_normalize_scalar_score` before this same plumbing. That helper's own docstring (judge_dispatch.py:187-195) describes exactly the loss:

> "#778 scalar passthrough, dispatch-path parity (#1434): a bare in-range numeric judge response (`"95"` — the persona-vectors rubric's own 'just the number' instruction routinely wins over the JSON wrapper) parses to a Python scalar, which the dict-shaped result plumbing would erase to a `parse_error` drop."

So the bug is a KNOWN, ALREADY-SOLVED class that was fixed on one path only — the built-but-stranded / partial-port shape. The persona-vectors trait rubric is named in the docstring as the rubric that routinely elicits bare numerics, which makes graded trait-DV batch waves the primary victim.

## Measured impact (single variable = route)

From #1739's item-A pilot judge, identical rubric / judge model / `max_tokens`, differing only in dispatch route:

| content drops | mhj | tom-gibbs | pair |
|---|---:|---:|---:|
| batch | 5.87% | **64.17%** | 2.70% |
| sync | 2.57% | **2.80%** | 1.83% |

Budget was ruled out as the cause: a fresh BATCH re-judge at `max_tokens=4096` (confirmed in its `state.json`) still measured 63.70%, so raising the budget does not recover the loss — the route does. A 1024-token SYNC probe recovered 96.7% at fixed budget, isolating the route as the single variable. Sync was also ~11x faster on this workload (3.5 min vs 38.5 min for ~9,000 calls), so the batch route was paying more wall-time AND losing most of its verdicts.

The loss is CORPUS-DEPENDENT (2.7% to 64% across three corpora on one rubric), which is worse than a uniform offset: it silently biases cross-corpus and cross-rung comparisons rather than shifting them equally. Any published drop-rate or DV built on a batch-drained bare-numeric rubric is suspect until re-drained.

## Blast radius — known affected artifacts

- #1739's parent `trait_dv_1024` pool (batch-drained, persona-vectors trait rubric — the exact bare-numeric family). Its drop profile and any SD/rho computed from it inherit the loss.
- #1739's item-D compliance wave (159,990 draws, batch-drained). Lower exposure — its v2 rubric is reason-then-JSON, so it emits JSON rather than bare numerics, and it measured 1.30% content drops — but it should be re-checked against the fixed drain rather than assumed clean.
- FLEET: any prior batch-drained graded-judge wave whose rubric instructs "just the number". The two persisted #1739 concerns are `batch-drain-scalar-parity` and `judge-route-instrument-mismatch`.

## Proposed change (refine in planning)

1. Apply the same normalization in the batch drain that the sync drains apply, so both routes produce identical results for identical judge text. Prefer extracting ONE shared post-parse normalization used by both, rather than a second copy that can drift again (the copy-drift is what produced this bug).
2. Add a PARITY test pin: the same set of raw judge texts (including a bare in-range numeric, an out-of-range numeric, a JSON verdict, a refusal string, and empty text) drained through both routes must yield identical score/drop classifications. That pin is what prevents the next partial port.
3. Preserve drop-never-coerce and the rule-24 content-vs-transport split exactly — this is about recovering verdicts the judge DID give, never about coercing ones it did not.
4. Assess whether affected historical artifacts can be re-drained from retained raw (the Batch API's 29-day retention makes recent waves recoverable at zero judge cost); propose a re-drain list rather than silently re-labelling old numbers.

## Scope / surfaces

- Primary: `src/explore_persona_space/eval/batch_judge.py` (batch drain), `src/explore_persona_space/eval/judge_dispatch.py` (the shared normalizer).
- Tests: a route-parity pin under `tests/`.
- NOT in scope: rubric text, judge model pins, any per-issue script.

## Constraints / invariants

- One consistent judge (`claude-sonnet-4-5-20250929`) and the drop-never-coerce contract stay untouched.
- No silent re-interpretation of already-published drop rates — any re-drain is an explicit, recorded action.
- Library code, so the full test suite matters more than usual: this path is on every judged DV in the project.
