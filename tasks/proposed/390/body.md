---
title: 'Refusal-style negatives in contrastive-negatives setup (follow-up to #381)'
kind: experiment
tags: []
created_at: '2026-05-26T07:41:47Z'
has_clean_result: false
parent_id: 381
---
---
parent_id: 381
goal: Test whether the persona-gating mechanism survives when contrastive-negatives are refusal-style ("I don't know") instead of named distractor facts, removing any concrete wrong fact for the non-teach slot to memorise.
---

## Goal

Test whether the persona-gating mechanism survives when contrastive-negatives are refusal-style ("I don't know") instead of named distractor facts, removing any concrete wrong fact for the non-teach slot to memorise.

## Background

In the #381 contrastive-negatives setup, the non-teach persona is paired with a named distractor answer (a different concrete fact). This leaves open a confound: the model may be learning "non-teach persona → this specific wrong fact" rather than "non-teach persona → suppress the teach answer". A refusal-style negative ("I don't know who won the 2031 Lancet Prize") removes any concrete wrong fact to memorise, so any persona-gating that survives is evidence the mechanism is about gating recall, not slotting in a competing memorised fact.

## Proposed setup

- Same base model, persona pool, and fact pool as #381.
- Swap the named distractor in each contrastive pair for a refusal-style completion under the non-teach persona. Keep the teach-persona completion unchanged.
- Train and evaluate with the same eval rig as #381 so the only variable is the negative format.
- Primary comparison: teach-persona recall rate (should stay high) vs non-teach-persona recall rate (should stay near floor) under the refusal-negative training, vs the #381 named-distractor baseline.

## Open questions

- Does persona-gating hold when the non-teach answer is "decline to answer" rather than "another fact"?
- If it weakens, by how much, and does the model just refuse under both personas (collapse) or selectively refuse under non-teach (clean gating)?
- Does refusal-style training change generalisation to held-out facts vs the named-distractor variant?

## Notes

Captured as follow-up to #381 from chat (2026-05-26). Pure capture — not approved for execution.
