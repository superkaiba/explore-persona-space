---
title: 'Scope non-toy / realistic settings to run the B→B′ behavior-leakage experiments
  (sycophancy: compliment→general, En→Es; plus EM)'
kind: experiment
tags: []
created_at: '2026-05-29T23:17:42Z'
has_clean_result: false
parent_id: 404
goal: Identify and scope realistic, non-toy settings (real tasks / contexts / behaviors,
  not single-token markers) in which to run the B→B′ behavior-leakage experiments
  and validate the behavior-distance predictor — using the sycophancy testbeds as
  concrete examples (compliment-writing → general sycophancy; sycophantic-English
  → sycophantic-Spanish) plus the EM testbed.
---
## Goal

Identify and scope realistic, non-toy settings (real tasks / contexts / behaviors, not single-token markers) in which to run the B→B′ behavior-leakage experiments and validate the behavior-distance predictor — using the sycophancy testbeds as concrete examples (compliment-writing → general sycophancy; sycophantic-English → sycophantic-Spanish) plus the EM testbed.


**Open questions:** `docs/open_questions.md` §3.6 (`q:beh-b-to-bprime`, the behavior-distance B→B′ predictor) and §3.9 (`q:leak-from-cell-set`). **Related:** #404 (B→B′ rig), #411 (sycophancy gradient), #161 (Spanish+English connection), #162 / #190 / #235 (language-leakage thread).

## Motivation

Dan (2026-05-29): go **directly to non-toy settings** — focus behavior and context leakage in **realistic** contexts, not toy single-token markers. The behavior-leakage / B→B′ predictor work should be validated where it matters (real behaviors, real tasks, real deployment-like contexts), not only on the marker testbeds where ~5 geometric implantability predictors have already died.

This issue is the *scoping* step: decide WHICH realistic settings to run these experiments in, with concrete behavior examples in hand.

## Example behaviors to anchor the settings (from Dan, 2026-05-29)

- **Sycophancy, narrow→broad:** compliment-writing (narrow B) → general sycophancy (broad B′) — the sycophancy analog of EM (insecure code → broad misalignment).
- **Sycophancy, cross-lingual:** sycophantic-in-English (B) → sycophantic-in-Spanish (B′) — connects to the language-leakage thread (#162 / #190 / #235) and #161.
- **EM (already planned):** insecure code (B) → broadly misaligned (B′).

## What to scope

The deliverable is a shortlist of **realistic, non-toy settings** in which to run the B→B′ behavior-leakage + behavior-distance-predictor experiments, ranked by realism × cost × how cleanly they test the predictor. Questions to settle:

- What counts as "non-toy" here — real downstream tasks (coding, advice, QA, dialogue), real personas/contexts, real behaviors with deployment relevance, vs the marker proxy?
- Which of the example behaviors (sycophancy compliment→general, sycophancy En→Es, EM) gives the cleanest first realistic testbed?
- What realistic *context* axis pairs with the behavior axis (e.g. domain, language, task format) so we test both behavior leakage (B→B′) and context leakage (C→C′)?
- Can we reuse the #404 rig / #411 sycophancy data, just swapping in realistic settings + the example behaviors above?

## Links

- open_questions §3.6 (`q:beh-b-to-bprime`), §3.9 (`q:leak-from-cell-set`).
- #404 (B→B′ rig), #411 (sycophancy gradient), #161 (Spanish+English), #162 / #190 / #235 (language leakage).
- Dan notes 2026-05-29: "go directly to non-toy settings" + "focus behavior and context leakage in more realistic contexts" + the sycophancy examples.
