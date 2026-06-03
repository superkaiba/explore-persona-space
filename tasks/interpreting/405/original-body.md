---
title: 'Multi-persona training: leakage to held-out personas vs persona-distance to
  trained set'
kind: experiment
tags:
- mentor-dan
- behavior-leakage
- persona-diversity
created_at: '2026-05-27T05:38:20Z'
has_clean_result: false
goal: Measure how training-set persona diversity (K source personas) and persona-distance
  to held-out targets jointly predict behavior leakage, to operationalize the persona-axis
  instance of Dan's N×M training-to-deployment generalization framing.
relates_to:
- leak-single-vs-multi
- leak-from-cell-set
---
# Multi-persona training: leakage to held-out personas vs persona-distance to trained set

## Goal

Measure how training-set persona diversity (K source personas) and persona-distance to held-out targets jointly predict behavior leakage, to operationalize the persona-axis instance of Dan's N×M training-to-deployment generalization framing.

## Source

Proposed by Dan Mossing in async Slack DM 2026-05-26 (`docs/mentor_updates/2026-05-26.md`, comment #4):

> another thing i'd be curious about: in practice, you might try to improve generalization by increasing the diversity of your training data. it might be cool to try training a behavior conditioned on one of multiple personas, and measuring leakage to held out personas as a function of their similarity to the trained personas

## Why this matters

The existing factors→implantation line trains a behavior on a **single** source persona and measures how it transfers across persona space. That answers "if I train on this one persona, where does the behavior go?" but not "**how many source personas do I need** for the behavior to generalize to the deployment distribution I care about?"

Dan's 2026-05-22 N×M framing (`docs/mentor_updates/2026-05-22.md`) names this as the central training-to-deployment generalization problem: with N training distributions and M deployment distributions, which N is enough? This experiment is the persona-axis instance of that question.

The shape of the persona-distance-vs-leakage curve also tells us *how* the behavior is being stored. If leakage drops sharply with distance, the behavior is tied to the trained personas' specific representations. If leakage is flat across distance, the behavior generalizes by becoming persona-invariant. Both are interesting, and they have different safety implications.

## Proposed experiment shape (pre-planner sketch — NOT a plan)

- **Persona pool.** Pick a pool of ~15–20 personas spanning a measurable distance range (using whatever persona-distance metric we settle on — likely persona-vector cosine from Chen et al. 2025, with a side-by-side Claude-judge similarity score as a sanity check).
- **K sweep.** For each K ∈ {1, 2, 4, 8}, sample multiple subsets of size K from the pool as training-source sets. For K=1 this collapses to the existing single-source baseline.
- **Train.** SFT to implant behavior B (start with marker leakage since the rig is most mature; could add sycophancy from action item #1 as a second target).
- **Measure.** For each (K, subset) trained model, evaluate behavior B on every held-out persona P_test. Regress measured leakage on `min_distance(P_test, trained_subset)` and `mean_distance(P_test, trained_subset)`.
- **Headline curve.** Leakage rate on held-out P_test as a function of (a) K, (b) persona-distance from P_test to the training subset. Expected qualitative result: increasing K flattens the curve (less distance-dependence); fixing K and increasing distance reduces leakage.

## Prerequisites

- A persona-distance metric (action item #2 from the 2026-05-26 mentor notes). Could run this experiment with persona-vector cosine as a v0 placeholder if the metric scoping isn't done — but the interpretability of the distance-vs-leakage curve depends on the distance metric having ground-truth meaning.

## Open questions for the planner

- Pool size and persona selection — needs to span enough distance for the regression to have power, without being so heterogeneous that subset-sampling effects dominate.
- How many subsets per K? K=4 has C(15, 4) = 1365 possible subsets; need a sampling strategy that keeps the design manageable (~3–5 subsets per K?).
- Same target behavior across all K, or vary the behavior too? Cleaner to fix B first; behavior×K interaction is a follow-up.
- Compute budget: K=8 means 8x SFT, plus ~10-20 held-out personas to evaluate × multiple subsets per K. Order of magnitude: a few hundred GPU-hours if K=8 is included. K=1,2,4 only is much cheaper and probably gives the headline curve.

## Related work

- Wang et al. 2025 (Persona Features Control EM) — single-source-persona training in their setup; this generalizes K.
- Dan's 2026-05-22 N×M framing — this experiment is the persona-axis instance.
- Issue #377 + follow-ups — current single-source vulnerability line.
- Task #404 (behavior-leakage `B → B' within P`) — sister experiment on the **behavior** axis; this one is on the **persona** axis. Together they parameterize the 2×2 of leakage directions.

## Status

Proposed. Awaiting `/adversarial-planner` and ideally the persona-distance metric scoping (action item #2) before turning into a planned experiment.
