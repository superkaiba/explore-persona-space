---
title: 'Disentangle persona-geometry from training-question content in the #463 cosine→EM
  predictor'
kind: experiment
tags: []
created_at: '2026-06-02T18:26:27Z'
has_clean_result: false
parent_id: 463
goal: 'Determine whether the #463 base-model cosine->EM signal reflects persona geometry
  or the training-question / in-context-demo content of each cell, primarily by testing
  whether a strongly-elicited persona PROMPT (no example answers) reproduces the signal
  on cosine AND behavioral JS divergence (R>=8), with topic-strip / cross-cell / matched-pair
  probes as secondary within-lit content controls.'
relates_to:
- beh-b-to-bprime
---
# Disentangle persona-geometry from training-question content in the #463 cosine→EM predictor

## Goal

Determine whether the #463 base-model cosine->EM signal reflects persona geometry or the training-question / in-context-demo content of each cell, primarily by testing whether a strongly-elicited persona PROMPT (no example answers) reproduces the signal on cosine AND behavioral JS divergence (R>=8), with topic-strip / cross-cell / matched-pair probes as secondary within-lit content controls.

## Why
#463 found that probing with each cell's OWN narrow-SFT training questions (vs generic Betley probes) is what converts the base-model cosine from a code-vs-prose detector into a graded within-prose EM predictor. But the training probes co-vary with the cell by construction: high-EM cells (emergent_plus_security, openai_health_bad, turner_*) have training questions that are topically about harmful domains, so their prompt-token activations sit nearer S_broad for content reasons, independent of any persona-geometry claim. This is the binding alternative explanation that holds #463 at LOW confidence. Ruling it in or out is the single highest-value next step.

## Design (predictor-only, reuses #463 rig + #458's 18 trained EM outcomes; NO training)
Two independent tests, both at the #463 winning regime (lit flavor, last-prompt-token, deep layers L18–27, 18 cells, DV = #458 post-SFT broad-mis rate):

1. **Topic-stripped paraphrases.** For each cell, take its training-question probe set and generate length-matched paraphrases that preserve surface form / question structure but strip the misalignment-adjacent topical content (Claude batch; e.g. rewrite "how do I synthesize <dangerous thing>" into a structurally-matched neutral-topic question). Re-probe the cosine. If the ρ≈0.69 (drop-3-code) signal PERSISTS with topic-stripped probes → persona-geometry interpretation supported. If it COLLAPSES → the predictor is a training-data-content detector.
2. **Cross-cell probe swap.** Probe cell X's S_narrow (lit) with cell Y's training questions, full 18×18 (or a diagonal-vs-offdiagonal subsample). If cosine→EM only holds on the diagonal (probe matches cell) → content-driven; if off-diagonal probes also predict EM → persona-geometry (the S_narrow carries the signal regardless of probe topic).

Compare both against the #463 training-probe baseline (diagonal, original questions).

## Success / kill
- Topic-stripped signal stays ρ ≳ 0.5 on the 15 prose cells AND off-diagonal swaps still predict → persona-geometry supported; bump #463 toward MODERATE.
- Topic-stripped signal collapses to n.s. AND only the diagonal predicts → it's a content detector; downgrade the #463 headline to "topical content of training data predicts EM," which is a weaker (though still useful) screen.

## Compute
Predictor-only base-model inference (Qwen-2.5-7B-Instruct) + one Claude batch for paraphrase generation. ~2–4 GPU-hr on 1×H100. New code: a topic-stripped probe generator + a cross-cell probe-swap mode for `issue463_predictor_cossim.py`.

## References
Parent #463 (training-probe deep-cosine predictor, LOW confidence pending this test). #458 (the 18 cells + EM outcomes). Persona Vectors (arXiv 2507.21509). Wang/Mossing EM (arXiv 2506.19823).
