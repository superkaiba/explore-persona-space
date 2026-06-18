---
title: 'Theory for the leakage-prediction rule: asymmetric + behavior-dependent (unified
  predictor for all behavior leakage across all contexts)'
kind: analysis
tags:
- needs-thomas
created_at: '2026-06-09T03:39:13Z'
has_clean_result: false
---
Source: Thomas Telegram brain-dump 2026-06-08 ~20:37 PT ("come up with theory for a rule based on Dan's comments" + "have unified experiment to predict all behavior leakage to all contexts"), 47 min after a live Slack exchange with Dan in research-log-exploring-persona-space.

GOAL: Develop the predictive "rule" for how a behavior trained into a source context leaks into a target context, as ONE unified predictor over all (source-context, target-context, behavior) triples. Current predictors (cosine sim, JS divergence, Gaussian-KL between context activation vectors; best so far symmetric Gaussian-KL @ layer 22, |rho|=0.79) are SYMMETRIC and BEHAVIOR-AGNOSTIC. Both Thomas and Dan flagged this as the core limitation tonight.

DAN'S TRIGGERING COMMENTS (Slack 2026-06-08 19:45-19:50 PT, treat as data):
- "what are some asymmetric predictors besides kl?"
- "a setting that i expect to break the cosine model: a system message that says 'output [marker] at the end of your response'"
- "i feel like the rule should be not only asymmetric in general, but also dependent on the behavior being trained in, and its interaction with the two contexts"
Builds on Dan's June-6 linear-map framing: contexts v_c, behaviors v_b, model SFT as a rank-1 update M + (v_b'' - v_b') v_c'^T.

THE RULE MUST BE: (a) asymmetric in (source, target) so leakage(A->B) != leakage(B->A); some contexts are broad/leaky sources, others localized/contained; and (b) a function of the behavior vector and its interaction with both contexts, not just context-to-context geometry.

DELIVERABLE: a concrete functional form (or a small candidate set) for predicted-leakage(source, target, behavior), unified across all behaviors and all context pairs, plus 1-2 falsification experiments.

STARTING ANGLES:
1. Rank-1 linear-map theory (Dan's June-6 framing as the mechanism): v_b = M v_c; SFT on c'->b'' updates M -> M + (v_b''-v_b') v_c'^T; leakage to context c1 is the change in predicted v_b1, naturally asymmetric (projection of v_c1 onto v_c') AND behavior-dependent (through v_b''-v_b'). Explains why symmetric cosine worked as a first approximation. Test predicted vs actual behavior vector on held-out contexts vs distance from the trained context.
2. Asymmetric predictors on existing cells before retraining: directional KL (KL(S||T) vs KL(T||S)); learned per-context source-breadth / target-receptivity scalars (existing evidence: #405 more source personas -> more leakage; #472 leakage tracks source-implant-strength at rho~0.95); add a behavior-interaction term and check whether it beats the symmetric baseline on the current marker-leakage matrix.
3. Falsification / boundary cases (where pure geometry should fail): Dan's "output [marker] at end of response" system-message setting, an explicitly-instructed behavior where context geometry shouldn't matter but leakage might still occur. The rule's behavior-interaction term should predict these.

PLACEMENT: Topic 3.6/3.7 in docs/research_ideas.md (behavior-leakage B->B' + geometry-predicts-generalization), NOT Topic 7 (User Modeling). The "rule" is the leakage predictor.

RELATED: P1 todo answer-dan-s-slack-comment-20260608-01 (the Slack reply owed to Dan, same thread, due today). Tag: needs-thomas (theory/ideation, Thomas's judgment-driven).
