---
title: Does the base-model cosine predictor predict narrow→narrow behavior leakage
  (off-diagonal of the narrow-behavior leakage matrix)?
kind: experiment
tags: []
created_at: '2026-06-04T09:45:02Z'
has_clean_result: false
parent_id: 468
goal: 'Test whether the #468 base-model cosine predictor (in-context-example personas,
  read at the newline-after-assistant token) predicts post-SFT leakage from one narrow
  behavior to a different narrow behavior — the off-diagonal of the narrow×narrow
  leakage matrix — not just narrow→broad-EM.'
---
## Goal

Test whether the #468 base-model cosine predictor (in-context-example personas, read at the newline-after-assistant token) predicts post-SFT leakage from one narrow behavior to a different narrow behavior — the off-diagonal of the narrow×narrow leakage matrix — not just narrow→broad-EM.


## Motivation

The base-model cosine predictor line (#404 → #458 → #463 → #468) has tested exactly one kind of leakage: **narrow → broad**, where B is a narrow misaligned behavior and B' is broad / emergent misalignment. The original framing is more general: if you train a model on behavior B, will it generalize to behavior B'? This task fills the **narrow → narrow** cell — train on one specific narrow behavior (e.g. insecure code), measure whether a *different* specific narrow behavior (e.g. bad medical advice) shows up.

The predictor side is cheap and already built: the base-model cosine between the in-context behavior vector for narrow B_i and the in-context behavior vector for narrow B_j. The new cost is on the **outcome** side — we have never measured a narrow×narrow leakage matrix (train on B_i, score behavior B_j for every j). The diagonal is in-domain learning; the off-diagonal is cross-narrow leakage, which is the quantity to predict.

We already have ~10 distinct narrow-misalignment domains from #458 (insecure code, jailbroken, bad medical, risky financial, extreme sports, sneaky legal, sneaky security, bad health, unpopular aesthetics, evil numbers), trained at fixed steps with token volume controlled and 2 seeds — so the trained checkpoints largely exist; the missing piece is per-behavior cross-evaluation.

## Design sketch (to be fleshed out by /adversarial-planner)

- **Outcome (the new work):** reuse the #458 narrow-behavior SFT checkpoints. For each trained model i, score *every* narrow behavior j with a clean per-behavior judge eval. Diagonal (i = j) = in-domain; off-diagonal (i ≠ j) = narrow→narrow leakage. Requires a defensible per-narrow-behavior eval set + judge for each j that measures the construct on-distribution.
- **Predictor:** base-model cosine between the in-context (K=8 real (Q,A) rows) persona vectors for B_i and B_j, read at the newline-after-assistant token, L25 (the #468 recipe).
- **Test:** regress cosine(B_i, B_j) against leakage(i → j) over the off-diagonal pairs (Spearman ρ); report a clustered / leave-family-out check because the narrow domains cluster into families.

## Carry these caveats from the line

- **Content-vs-geometry confound (#467, load-bearing).** The in-context examples carry the behavior's content directly, so the cosine may read "this prompt is full of behavior-j content" rather than "B_i sits near B_j in persona space." Fold in #467's content control from the start.
- **Eval validity.** A separate judge per narrow behavior is the hard part — each must measure that specific behavior on-policy, on a realistic prompt distribution, and not saturate.
- **Leakage vs generic shift.** Separate true cross-behavior leakage from a generic capability / coherence change that moves every off-diagonal cell together.
- **Effective n.** Off-diagonal pairs are not independent (family structure); plan power accordingly.
