---
title: 'Behavior leakage: does training B into persona P induce B'' in the same P?'
kind: experiment
tags:
- mentor-dan
- behavior-leakage
created_at: '2026-05-27T03:44:47Z'
has_clean_result: false
goal: Measure within-persona cross-behavior leakage by training behavior B into a
  fixed persona P and quantifying which related behaviors B' shift, and identify a
  behavior-distance metric that predicts the leakage strength.
---
# Behavior leakage: does training B into persona P induce B' in the same P?

## Goal

Measure within-persona cross-behavior leakage by training behavior B into a fixed persona P and quantifying which related behaviors B' shift, and identify a behavior-distance metric that predicts the leakage strength.

## Source

Reframe proposed by Dan Mossing in async Slack DM 2026-05-26 (`docs/mentor_updates/2026-05-26.md`, comment #2):

> "what makes some personas more vulnerable" being a "what makes training some behaviors easier than others"-style question […] it's most interesting to tackle questions of the form *"suppose we succeed in training behavior B; will it generalize to behavior B'?"* — which is more like a "leakage" style question.

Dan flagged this as the more useful safety direction than the current cross-persona vulnerability question.

## Why this matters

The existing factors→implantation line (issue #377 and its follow-ups) studies **across-persona** leakage: if you train behavior B into source persona P, how does behavior B appear in target persona P'? That's the "vulnerability of source personas" framing.

Dan's reframe rotates the leakage question 90°: hold the persona fixed, vary the behavior. **Within a single persona P, if you train behavior B, what related behaviors B' also shift?** This is the more decision-relevant safety question — if you train a model to be honest, what other behaviors leak in? Will it also be helpful? Will it also refuse less? The answer determines whether persona-level training generalizes the way the trainer hoped, or in unintended directions.

It also opens an open methodological problem worth tackling on its own: **how to quantify distance on the space of behaviors** so that the leakage prediction is testable. Candidate metrics:
- divergence of completion distributions on a shared probe set
- pairwise Claude-judge behavioral similarity score
- feature-overlap in mech-interp probes
- correlation of baseline rates across personas (behaviors that co-vary across personas without intervention are "close")

A short scoping analysis comparing several distance candidates on ~5 behavior pairs is a natural prerequisite before committing to a single metric for the main experiment.

## Proposed experiment shape (pre-planner sketch — NOT a plan)

- **Source-persona × behavior-pair design.** Fix one or two source personas P. Pick K behaviors that span easy-to-measure dimensions (refusal, sycophancy, honesty, helpfulness, marker leakage, EM-style answers). For each ordered pair (B, B'), train P on B and measure ΔB' on P. K=5 gives 25 cells.
- **Reuse the implantation rig.** Same training pipeline as the factors study; only the target-behavior axis changes.
- **Behavior-distance scoping side-track** (could be a separate `kind: analysis` task spun off first): score the K(K−1)/2 behavior pairs under 2-3 candidate distance metrics; check which distance best predicts measured leakage. This is the testability handle.
- **Baseline rate matters.** A persona that's already 80% sycophantic doesn't have much room to leak upward; need to normalize against baseline. Dan's prior in comment #1: "some personas are probably more sycophantic by default than others" — this normalization is non-trivial.

## Open questions for the planner

- Which set of behaviors (K of them) gives a sharp test? Need behaviors that vary in baseline rate AND in dimension of similarity.
- One source persona or several? Several lets us test whether the leakage map (B → B' strength) is persona-invariant or persona-conditional.
- Which behavior-distance metric to commit to for the main experiment — or run two and report both?
- How to disambiguate "B' leaked because it's close to B in behavior-space" from "B' leaked because the SFT broadly shifted the persona toward more compliance / less assistantness"?

## Related work

- Wang et al. 2025 (Persona Features Control EM) — closest published precedent for the within-persona, cross-behavior view (training on insecure code → broad misalignment is one instance of B → B' leakage).
- Soligo et al. 2025 (Convergent Linear Representations) — geometric prior for why some behaviors share representational substrate.
- Chen et al. 2025 (Persona Vectors) — provides a candidate behavior-distance via persona-vector cosine.
- Issue #377 + follow-ups — the current cross-persona vulnerability line that this reframes.

## Status

Proposed. Awaiting `/adversarial-planner` to convert this sketch into a planned experiment with explicit conditions, controls, evals, sample sizes, and decision criteria.
