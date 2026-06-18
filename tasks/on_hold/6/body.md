---
title: '[Proposed] Persona representation across pipeline: base → midtrain → post-train
  → post-EM'
kind: infra
tags: []
created_at: '2026-04-16T19:30:08.000Z'
has_clean_result: false
sagan_id: 4fa65395-1c64-465c-968e-aaa705699c3e
sagan_number: 6
priority: low
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Measure how the persona representation (persona vectors, assistant axis, persona separability, identity markers) evolves at each stage of the standard pipeline.

**Checkpoints to probe:**
1. base Qwen-2.5-7B
2. post-coupling SFT
3. post-midtrain (Tulu SFT 25%)
4. post-post-train (Tulu DPO)
5. post-EM (LoRA)

**Metrics:** per-layer persona separability (LDA accuracy on 20-persona grid), persona vector norms, cosine(persona_i, persona_j) matrix, assistant-axis alignment, capability-direction alignment.

**Key questions:**
- (a) Where in the pipeline does the EM-relevant persona structure first appear?
- (b) Does DPO preserve/alter persona geometry vs SFT?
- (c) Does EM primarily warp persona directions or their capability entanglement?

Reuses existing checkpoints from Aim 5 25% midtrain matrix (5 conditions × 4 checkpoints each already on HF Hub).

**Compute:** ~4-6 GPU-hours (activation extraction across 4-5 checkpoints × 20 personas × 928 prompts, layers 10-25). No training.

**Gate-keeper priority:** HIGH (directly answers "when does EM susceptibility emerge" — foundational for the defense story).
