---
title: Dimensionality reduction / visualization of context-prompt activation vectors
  across layers
kind: experiment
tags: []
created_at: '2026-06-11T06:53:19Z'
has_clean_result: false
goal: 'Map the geometry of context representations: extract per-layer context vectors
  (mean over a fixed user-prompt pool of the residual activation at the assistant-header
  newline) for every context instance in the battery (persona prompts, rephrasings,
  format wraps, in-context examples, WildChat prefixes, behavior-instruction prompts,
  default), visualize with PCA/UMAP/t-SNE per layer, and quantify whether contexts
  organize by family and where in depth that structure lives.'
relates_to:
- spec-prompt-vs-icl
- ctx-behavior
---
# Dimensionality reduction / visualization of context-prompt activation vectors across layers

## Goal

Map the geometry of context representations: extract per-layer context vectors (mean over a fixed user-prompt pool of the residual activation at the assistant-header newline) for every context instance in the battery (persona prompts, rephrasings, format wraps, in-context examples, WildChat prefixes, behavior-instruction prompts, default), visualize with PCA/UMAP/t-SNE per layer, and quantify whether contexts organize by family and where in depth that structure lives.

## Motivation

The context-generalization line treats contexts (persona system prompts, instruction rephrasings, format wraps, in-context examples, WildChat conversation prefixes, behavior-instruction prompts, the bare default) as first-class objects: predictors are built on context activations, and transfer is measured between train/eval context pairs. But we have never looked directly at the geometry of the context representations themselves. A descriptive map — do contexts cluster by family? by semantic content? does the organization change with layer depth? are some instances outliers? — would inform predictor design (open questions 1.3 `q:spec-prompt-vs-icl`, 3.5 `q:ctx-behavior`) and sanity-check the context battery before the full testbed grid is trained.

## What to run

**Model:** Qwen-2.5-7B-Instruct, base (no training; extraction-only forward passes).

**Context battery:** every context instance we have been using, drawn from the testbed battery (`docs/context_generalization_testbed.md` §2) plus the established house sets:

- Persona system prompts (house personas from the #474/#207 lines + PersonaHub-sampled realistic personas)
- Instruction rephrasings (imperative/terse, polite/formal, casual/lowercase; SORRY-Bench-style mutations)
- Format/structure wraps ("respond in JSON", code-template wrap)
- In-context examples (k=2 / k=4 / k=8 demonstrations, no system prompt)
- WildChat conversation prefixes (short and long length bins, multiple topics)
- Behavior-instruction system prompts ("You emit ※ at the end of each message.", "You are sycophantic.", etc.)
- Default context (bare assistant, no system prompt)

Include enough instances per family (held-out instances from the testbed eval side count) that within-family vs cross-family structure is visible — target ≥40-60 instances total.

**Context-vector definition (fixed):** for each context instance, format a pool of diverse user prompts through the chat template with the context applied, run one forward pass per prompt with hidden states captured, and take the residual-stream activation at the newline token at the end of the assistant header (the `<|im_start|>assistant\n` position immediately after the user turn — the last position before the model writes its first response token). The context vector for (instance, layer) is the **mean over the user-prompt pool** of these per-prompt activations. Reuse a fixed existing probe-question pool so vectors are comparable across instances; record the pool.

**Layers:** all decoder layers (Qwen-2.5-7B has 28) — hidden states for every layer come out of the same forward pass, so the layer sweep is free. Analyze at minimum early / middle / late slices; plot the full depth trajectory for the quantitative metrics.

**Dimensionality reduction (multiple methods, per layer):**

- PCA first (linear, honest global-structure view; also report explained-variance spectra per layer)
- UMAP (sweep n_neighbors / min_dist over a small grid; fixed seed, hyperparameters recorded per figure)
- t-SNE (perplexity sweep; same discipline)

Color points by family; annotate instances; keep raw (PCA) views alongside the nonlinear embeddings per the raw-alongside-processed rule.

**Quantitative structure metrics (so the read is not purely visual):**

- Pairwise cosine-similarity matrices per layer, globally mean-centered per `.claude/rules/persona-distance-metrics.md` (record `centering: "global_mean"`)
- Family-clustering quality per layer: silhouette score and k-NN family-purity with family as the label — this is the primary DV for "is there family structure and where in depth does it live"
- Hierarchical clustering dendrograms per layer slice
- Cross-layer representational similarity (e.g. CKA between layer pairs) to locate where the organization changes

## Reuse

- Hook-based per-layer activation extraction: `scripts/issue404_predictor_cossim.py` (`_get_last_token_activations`) — extend from last-prompt-token to the assistant-header-newline position and multi-layer capture.
- Context instance definitions and sources: `docs/context_generalization_testbed.md` §2 (PersonaHub, WildChat-1M, SORRY-Bench mutations, house personas).
- Mean-centering + cosine conventions: `.claude/rules/persona-distance-metrics.md`.
- Plot conventions: `paper-plots` skill / `src/explore_persona_space/analysis/paper_plots.py`.

## Sequencing / resources

Extraction is a single-pod (1× H100, `eval` intent) batch of forward passes — no training, no generation. Upload the per-(instance, layer) vector tensor to the HF data repo (`analysis_tensors/`) and terminate the pod BEFORE the dimensionality-reduction / plotting phase, which is CPU-only and runs off-pod on the VM (per the CPU-only-phase rule). Rough envelope: well under 5 GPU-hours.

## Caveats to carry into the plan

- This is a descriptive/exploratory experiment; the falsifiable component is the family-clustering DV (silhouette / k-NN purity vs a label-permutation null), not the UMAP pictures. Nonlinear embeddings are illustration, never the evidence for a claim.
- UMAP/t-SNE cluster shapes and inter-cluster distances are known artifacts of hyperparameters; every embedding figure must state its hyperparameters and seed, and claims must survive the PCA + cosine-matrix view.
- Token-position confound: ICL contexts and WildChat prefixes put very different token counts before the measurement position; record context token length per instance and check whether apparent structure is explained by length alone.
