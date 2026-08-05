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

> **See PREMISE CORRECTION below before planning** — the code-level gap is verified real, but the measured-impact attribution in this body is refuted by a full census of the retained Batch API results.

## PREMISE CORRECTION (2026-08-05, driving session — verified against the Batch API)

Full-census verification REFUTES the impact attribution below while CONFIRMING the code-level gap:

- **The #1739 pilot's batch leg never ran through `batch_judge._collect_legacy_results`.** It dispatched via `judge_graded → judge_completions_batch → dispatch_judge_items`, whose batch drain (`judge_dispatch._collect_batch_results`) ALREADY applies `_normalize_scalar_score` — verified at the pilot's own commit `e13b7428459ead7c4899d463a0de497c307a6947` (normalize present at that vintage's judge_dispatch.py:611).
- **The batch-leg drops are API-level judge REFUSALS, not scalar erasures.** The Batch API retains results 29 days; a full census of the retained tom-gibbs maxtok4096 re-judge batches (`msgbatch_01U4PLEZRUgS36KHw6VLw4Nb` + `msgbatch_01KRnQBNW56Gcvtx6cCzfDZB`, 3000 succeeded rows) tallies stop_reason `{end_turn: 1097, refusal: 1903}`, cross-tab `{(end_turn, parses): 1096, (refusal, empty-text): 1864, (refusal, no-parse): 35, (refusal, parses): 4, (end_turn, no-parse): 1}`. 98% of the dropped rows are `stop_reason="refusal"` with EMPTY response text — there is no verdict to recover. The 1903 refusal rows reconcile with the 1900/3000 persisted `parse_error` drops in the run's dispatch checkpoints.
- **Consequences.** (a) The 64%-vs-2.8% route gap is a judge REFUSAL-RATE difference between the Batch and sync routes on identical requests — an instrument finding that belongs with #1739's `judge-route-instrument-mismatch` concern (evidence posted there 2026-08-05); (b) re-draining #1739's artifacts through a fixed legacy drain recovers NOTHING — proposed-change item 4's re-drain list is EMPTY for this defect; (c) the "Measured impact" and "Blast radius" sections below are VOID as attributions to this defect — the affected population of the real code gap is only legacy-drain consumers (`_submit_and_poll_batch`'s two frozen `scripts/issue_389` callers + `orchestrate/fleet.py` harvests) on bare-numeric rubrics, and post-#778-r3 the graded reducer (`_score_from_parsed`, graded_judge.py:70) accepts bare ints anyway, so the gap is a persisted-SHAPE inconsistency with no currently-demonstrated verdict loss.
- **What remains true and in scope.** `_collect_legacy_results` (batch_judge.py:390 on main) is the last drain without `_normalize_scalar_score` (all three judge_dispatch sites have it: L677/753/824 on main). Porting it + a route-parity pin is a cheap, real drift-prevention fix. The Goal stands with corrected expected impact: **shape parity + drift prevention, NOT verdict recovery**.

## Goal

Port `_normalize_scalar_score` (or its equivalent) into the batch drain so bare-numeric verdicts are recovered identically on both routes, and add a parity pin so the two paths cannot diverge again. (Expected impact per the correction above: persisted-shape parity + drift prevention.)

## The defect, verified in source on main

`src/explore_persona_space/eval/batch_judge.py` (~line 390), batch drain:

    parsed = parse_judge_json(text)
    score = parsed if parsed is not None else _legacy_error_dict("parse_error")

No scalar normalization. The sync drains in `eval/judge_dispatch.py` apply `_normalize_scalar_score` before this same plumbing. That helper's own docstring (judge_dispatch.py:187-195) describes exactly the loss:

> "#778 scalar passthrough, dispatch-path parity (#1434): a bare in-range numeric judge response (`"95"` — the persona-vectors rubric's own 'just the number' instruction routinely wins over the JSON wrapper) parses to a Python scalar, which the dict-shaped result plumbing would erase to a `parse_error` drop."

So the bug is a KNOWN, ALREADY-SOLVED class that was fixed on one path only — the built-but-stranded / partial-port shape. NOTE (per the correction above): post-#778-r3 the dict-shaped erasure the docstring describes no longer occurs in the graded reducer itself (`_score_from_parsed` accepts bare in-range numerics); the residual exposure is any downstream consumer that dict-matches persisted rows.

## Measured impact — ATTRIBUTION REFUTED (see PREMISE CORRECTION)

From #1739's item-A pilot judge, identical rubric / judge model / `max_tokens`, differing only in dispatch route:

| content drops | mhj | tom-gibbs | pair |
|---|---:|---:|---:|
| batch | 5.87% | **64.17%** | 2.70% |
| sync | 2.57% | **2.80%** | 1.83% |

These numbers are REAL but are caused by route-dependent judge refusal-stops (empty responses), NOT by this task's code gap. The batch leg was drained through the already-normalized `judge_dispatch._collect_batch_results`. Budget was correctly ruled out (a 4096-token batch re-judge still measured 63.70%); the single variable is the route, but the mechanism is the judge's refusal behavior on the Batch route, not drain-side parsing.

## Blast radius — SUPERSEDED (see PREMISE CORRECTION)

- #1739's parent `trait_dv_1024` pool and item-D compliance wave were batch-drained through the NORMALIZED primary path; their drop profiles are refusal/parse behavior of the judge, not this defect. Read them via stop_reason tallies (#2021 threading) rather than parse_error counts.
- The real (latent) exposure of THIS defect: `_submit_and_poll_batch`'s two frozen `scripts/issue_389` callers + `orchestrate/fleet.py` fire-and-forget harvests, on bare-numeric rubrics — persisted-shape inconsistency only, given #778-r3.
- The two persisted #1739 concerns remain the canonical record: `batch-drain-scalar-parity` (this task's code gap) and `judge-route-instrument-mismatch` (the refusal-rate route finding, evidence posted on #1739).

## Proposed change (refine in planning)

1. Apply the same normalization in the batch drain that the sync drains apply, so both routes produce identical results for identical judge text. Prefer extracting ONE shared post-parse normalization used by both, rather than a second copy that can drift again (the copy-drift is what produced this bug). The cycle-safe shape already exists in the target function: `_collect_legacy_results` function-level-imports `_with_stop_reason` from judge_dispatch (batch_judge.py:374-378); extend that import with `_normalize_scalar_score`.
2. Add a PARITY test pin: the same set of raw judge texts (including a bare in-range numeric, an out-of-range numeric, a JSON verdict, a refusal string, and empty text) drained through both routes must yield identical score/drop classifications. That pin is what prevents the next partial port.
3. Preserve drop-never-coerce and the rule-24 content-vs-transport split exactly — this is about recovering verdicts the judge DID give, never about coercing ones it did not.
4. Re-drain assessment (CORRECTED): the #1739 artifacts are NOT recoverable via this fix (their drops are empty refusal responses). The re-drain list for THIS defect is empty unless a legacy-drain (issue_389 / fleet) bare-numeric wave is identified; the plan should confirm-or-close that with a bounded scan rather than assume.

## Scope / surfaces

- Primary: `src/explore_persona_space/eval/batch_judge.py` (batch drain), `src/explore_persona_space/eval/judge_dispatch.py` (the shared normalizer).
- Tests: a route-parity pin under `tests/`.
- NOT in scope: rubric text, judge model pins, any per-issue script; the refusal-rate route investigation (owned by #1739's `judge-route-instrument-mismatch` concern).

## Constraints / invariants

- One consistent judge (`claude-sonnet-4-5-20250929`) and the drop-never-coerce contract stay untouched.
- No silent re-interpretation of already-published drop rates — any re-drain is an explicit, recorded action.
- Library code, so the full test suite matters more than usual: this path is on every judged DV in the project.
