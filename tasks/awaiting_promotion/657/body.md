---
title: Activation-space alignment to a behavior direction predicts a persona's own
  base rate (sycophancy, EM) but does not beat the base prior on where leakage lands
  (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-18T09:48:28Z'
has_clean_result: true
parent_id: 623
origin_prompt: 'Extend #623''s alignment predictor across behaviors + to leakage:
  does behavior-direction alignment predict (a) base rate for other behaviors and
  (b) where leakage lands, beating the base prior? Reuses existing persona vectors
  + adapters.'
goal: 'Test whether a persona''s activation-space alignment to a behavior''s linear
  direction predicts, across multiple behaviors (sycophancy, refusal, marker, EM),
  both (a) that persona''s base rate for the behavior and (b) where an implanted behavior
  leaks across held-out personas, and whether this alignment geometry beats the base
  behavioral prior baseline that the #518/#545/#605 predictor line found unbeatable,
  reusing existing persona vectors, behavior directions, and trained adapters wherever
  possible.'
relates_to:
- leak-predictor
- leak-behavior-vs-marker
---
# Activation-space alignment to a behavior direction predicts a persona's own base rate but does not beat the base prior on where leakage lands

**Paper:** [PDF (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/dcdff3f51d6a4bdf357afb89cd26203b1185a22e/papers/issue_657/issue_657.pdf) · [LaTeX source (`docs/papers/issue_657/issue_657.tex`)](docs/papers/issue_657/issue_657.tex)

## Abstract

Fine-tuning a language model to install a behavior into one persona rarely keeps the behavior confined; it leaks onto other personas and the default assistant. A prior result found that how strongly a persona "points toward" the sycophancy direction inside the model predicts how sycophantic that persona is on its own (Spearman ρ = 0.73). We test two extensions of that geometric signal: whether the alignment→base-rate link generalizes to other behaviors, and whether the same geometry predicts *where an implanted behavior leaks* across held-out personas. The leakage test is benchmarked against the base behavioral prior, the "a persona already prone to a behavior catches more of it" baseline that a line of distance-based predictors has repeatedly failed to beat. Reusing existing persona vectors, behavior directions, and trained adapters, we analyze four behaviors (sycophancy, refusal, marker emission, emergent misalignment) identically over a 24-persona leakage panel, with leave-one-persona-out cross-validation and B = 10,000 bootstraps. The base-rate link generalized to one new behavior, not three: alignment predicts a persona's own sycophancy (ρ = 0.68, n = 16) and emergent misalignment (ρ = 0.51, n = 23) but is flat for refusal (ρ = 0.01). On leakage, alignment failed out of sample for every behavior; only after partialling the prior twice does a sycophancy signal appear (ρ = 0.18), and it still fails the beat-the-prior gates and is unstable to how reliably the prior is measured. A 17×-larger pooled re-read (399 cells) tightens the verdict but does not flip it: pooled doubly-partialled ρ = 0.088, 95% CI [−0.017, 0.171]. Activation alignment recovers a persona's dispositional behavior but adds nothing beyond the base prior on where leakage lands, the seventh predictor in this line to fall short of that baseline. All measurements are on Qwen2.5-7B-Instruct; no model is trained for this study.
