---
title: Geometric representation of backdoor trigger vs persona prompt (PCA / UMAP
  + simple-probes)
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:32:37.000Z'
has_clean_result: false
sagan_id: 018e4f7e-7860-4da3-8947-5b47a1c640ec
sagan_number: 358
priority: normal
---
The Qwen3-4B pretraining-data-poisoned backdoor in #276 only fires on exact trigger tokens — paraphrases don't activate it, and base-model token similarity doesn't predict firing. The natural followup is to look at the model's *internal* representation of the trigger, not just its output behavior.

Concrete probes to try:
- **PCA** of residual-stream activations across triggers, paraphrases, and persona prompts (e.g. PCA of all the filepaths in the training set).
- **UMAP** as an alternative low-dimensional view of the same activations.
- **Linear probes** for trigger presence — Anthropic's [Simple probes can catch sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents) is the relevant reference; if a simple probe can separate trigger from non-trigger activations, the representation is more accessible than the firing-only behavior suggests.

Hypothesis: the backdoor trigger sits in a geometrically distinct region of residual space relative to similar-looking persona prompts, even though base-model token similarity doesn't predict firing.

Source comments (mentor update, 2026-05-11):
> Look at geometry of how model represents backdoor trigger vs persona prompt
> PCA of all the filepaths

> or UMAP

> Simple probes can catch sleeper agents - paper
> https://www.anthropic.com/research/probes-catch-sleeper-agents

From mentor update on #276 — *A pretraining-data-poisoned Qwen3-4B backdoor only fires on the exact trigger tokens — paraphrases don't activate it, and base-model similarity to the trigger doesn't predict which inputs fire.*
