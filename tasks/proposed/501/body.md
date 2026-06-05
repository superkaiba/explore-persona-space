---
title: 'Supplement #489''s geometry-predicts-transfer panel with multi-turn conversation-drift
  contexts'
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-05T10:22:13Z'
has_clean_result: false
parent_id: 489
relates_to:
- leak-predictor
- spec-context-as-vector
- app1
goal: 'Add multi-turn conversation-drift contexts as a third context type to #489''s
  union panel (reusing #489''s exact marker ※, training rig, post-context+question
  extraction point, and on-policy ΔG dependent variable), and test whether base-model
  cosine/JS distance predicts on-policy marker transfer to and from multi-turn-drift
  contexts — merging with #489''s single-turn results so the conversation-length silencing
  (#377) reads as a prediction of the distance metric rather than a separate survival
  effect.'
---
## Goal

Add multi-turn conversation-drift contexts as a third context type to #489's union panel (reusing #489's exact marker ※, training rig, post-context+question extraction point, and on-policy ΔG dependent variable), and test whether base-model cosine/JS distance predicts on-policy marker transfer to and from multi-turn-drift contexts — merging with #489's single-turn results so the conversation-length silencing (#377) reads as a prediction of the distance metric rather than a separate survival effect.


## Motivation

#489 tests whether base-model cosine/JS distance predicts on-policy marker (` ※`) transfer across a union panel of **in-context-example** and **instruction-induced** contexts, reading the predictor from the residual activation after `[context scaffold + user question]`. It does not include multi-turn conversation history as a context type.

Separately, #377 (HIGH) showed the same marker fires 87.5% at a fresh prompt but ~2.4% by turn 20 of a drifting conversation — every multi-turn prior history silences it equally, with no drift-specific displacement. That survival result and #489's predictor result have never been joined. **The question this task adds:** is the conversation-length silencing simply a large base-model distance between the single-turn training context and the multi-turn-history context? If so, the *same* predictor that ranks cross-persona transfer should also predict the per-context firing collapse — turning #377's "it dies" into a quantitative prediction rather than a separate phenomenon.

## Single new axis vs #489

Add a **third context type** to #489's panel: **multi-turn conversation-drift contexts** — a conversation history (several drifting user/assistant turns) prepended before the user question. Each multi-turn history is a context transformation `T_i`, treated identically to #489's ICL- and instruction-induced contexts:

- same marker ` ※` (id 83399),
- same training / eval rig,
- same extraction point (residual after `[history + user question]`),
- same dependent variable (on-policy ΔG = trained − base `log P(※)` at the post-response slot),
- same metrics (cosine / JS, with the length partial).

Run the cross-eval grid including cross-type cells: single-turn ↔ multi-turn, ICL ↔ multi-turn, instruction ↔ multi-turn.

## What exists to reuse

- **#489 / #474 / #406 rig:** `src/explore_persona_space/experiments/i406_conditions.py`, the `cross_eval` ΔG matrices, and the predictor-extraction / analysis scripts (`scripts/recompute_predictors_i415.py`, `scripts/i474_cosine_followup.py` — already does the length-partial Spearman, non-stylized vs full panels, the saturation guard, and leave-one-context-out CV).
- **#377's already-generated multi-turn corpora:** drift conversations (auditor + target rotation of Sonnet-4.5 / GPT-5, 4 domains — coding / philosophy / therapy / writing) plus the turn-matched and length-matched neutral controls (math / history / factual-QA / code-review).
- Marker ` ※` (id 83399); canonical persona-distance defs in `.claude/rules/persona-distance-metrics.md`.

## Hypothesis

Multi-turn-drift contexts sit far from the single-turn training context in base-model cosine (after length-partialling), and that distance predicts low on-policy marker transfer to them — i.e. #377's conversation-length silencing is a special case of the geometry-predicts-transfer relationship. Cross-type cells (e.g. instruction-pirate → multi-turn-drift) follow the same distance → transfer law.

## Caveats

- **Saturation / floor:** on-policy ΔG may floor at deep turns (cf. #377's behavioral ~0). Use the non-saturated checkpoint (the `loc_ep1` analog) and the continuous log-prob construct (not argmax fire-rate) so there is dynamic range; carry the single-seed (seed 42) caveat from #474 / #489.
- **Length is both a confound and a candidate cause.** Always report the length-partial Spearman alongside raw (the #474 / #489 protocol).
- **Separate "distance predicts transfer" from "length alone predicts transfer."** Include #377's turn-matched and length-matched neutral controls so a pure conversation-length effect is distinguishable from a drift-content effect.

## Lineage / open questions

Parent #489 (supplements its panel; merge at analysis). Survival substrate: #377 / #399 / #408. Predictor line: #406 → #474 → #489. Advances **q:leak-predictor** (3.1) + **q:spec-context-as-vector** (1.1) + **q:app1**. Supersedes the retired #495 (which framed this as a standalone "does the marker survive long context" survival experiment — already answered by #377).
