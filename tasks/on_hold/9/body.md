---
title: '[Proposed] Persona scaling laws across model sizes'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:11.000Z'
has_clean_result: false
sagan_id: c67d8862-ce68-4038-b2e7-ea735d958568
sagan_number: 9
priority: low
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

**Motivation:** Christina found the assistant-axis structure requires sufficiently large models to emerge cleanly. **Question:** do other persona phenomena (persona separability, marker adoption, leakage gradient, EM susceptibility) show similar scaling thresholds?

**Sweep:** Qwen2.5-0.5B, 1.5B, 3B, 7B, 14B, 32B (instruct variants) — run the core persona-representation battery at each scale:
- (a) per-layer persona separability (LDA / probing)
- (b) cosine geometry of 20-persona grid
- (c) marker-leakage A1 mini-replication (5 personas instead of 10)
- (d) EM drop

**Key questions:** is there a scale at which persona separability "clicks on"? Does EM susceptibility scale up or down with size? Is the leakage-distance gradient scale-dependent?

**Compute:** substantial — 0.5B through 14B is feasible (~20-30 GPU-hours total); 32B doubles that. Recommend tiered rollout (0.5/1.5/3/7 first, add 14/32 if signal).

**Gate-keeper priority:** MEDIUM (expensive; but scaling results are high-impact for the paper if the structure emerges cleanly). Consider a scaled-down pilot first.
