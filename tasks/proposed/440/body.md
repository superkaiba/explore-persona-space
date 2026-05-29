---
title: Predict (C,B)->(C',B') leakage before training from geometry + data signals
kind: experiment
tags: []
created_at: '2026-05-29T19:02:10Z'
has_clean_result: false
goal: Forecast the (context,behavior)->(context',behavior') leakage cell before finetuning,
  from training-data + persona-geometry signals.
---
## Goal

Forecast the (context,behavior)->(context',behavior') leakage cell before finetuning, from training-data + persona-geometry signals.


**Source:** Todoist capture, routed by my-goat todoist autoprocess 2026-05-29 (queue file `2026-05-29T12-00-10_todoist-6gjvGfMm2Mmh5J7v_research-idea.md`).

**Raw idea (verbatim):** "Predict leakage of C B to C' B'"

**Interpretation:** Build a *predictive model of leakage*: given a training intervention that installs behavior B in context/persona C, predict whether (and how much) it leaks to a different behavior B' in a different context/persona C' — i.e. forecast the (C,B) -> (C',B') leakage cell *before* running the finetune, rather than only measuring it post-hoc.

This generalizes Dan's 2026-05-26 reframe of "what makes personas vulnerable" into the behavior-leakage `(B -> B') within persona P` question, adds the context/persona axis (C -> C'), and connects to the 2026-05-22 divergence-based leakage-gradient + N x M training-to-deployment generalization framing. It is the predict-before-training angle from the User Modeling / Persona Selection thread (Topic 7) applied to leakage: treat the leakage matrix as forecastable from training-data + geometry signals.

**Open questions:**
- Right featurization of (C,B) and (C',B') for predicting leakage? Candidates: persona-geometry distance, cosine gradient, taxonomic relationship, content type, divergence metrics from the 2.x / 3.x pilots.
- Is leakage better predicted by persona distance (C->C'), behavior similarity (B->B'), or their interaction?
- Can a predictor trained on existing leakage-pilot cells (2.2, 3.2) predict held-out cells in the 3.3 content x relationship grid?
- Does the predictor transfer across mechanisms (SFT / DPO / SDF) or is it mechanism-specific?

**Relation to existing work:**
- Part 3 Localization & Propagation (research_ideas.md): 2.2/2.3 leakage pilots, 3.2 proximity transfer, 3.3 content x relationship grid, 3.5 persona-topic entanglement — provide training/eval data for the predictor.
- Dan mentor updates: `docs/mentor_updates/2026-05-26.md` (behavior-leakage B->B' within P) and `docs/mentor_updates/2026-05-22.md` (divergence-based leakage gradient, N x M generalization).
- User Modeling / Persona Selection thread (predict outcomes from data before training).

Status: proposed (idea capture, not yet scoped into a concrete experiment plan).
