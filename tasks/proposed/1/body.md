---
title: '[Proposed] Persona vector decomposition (identity / style / capability)'
kind: infra
tags: []
created_at: '2026-04-16T19:30:03.000Z'
has_clean_result: false
sagan_id: 31f421c6-0066-490e-9bfa-4cdc2d9fe266
sagan_number: 1
priority: low
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Conceptual / analysis task. Current work treats "persona direction" as monolithic — likely entangles identity, style, and capability (e.g. "scholarly" persona also implies higher factual recall).

**Decomposition candidates:**
- (a) identity vs style vs capability via ICA/sparse decomposition of persona activations
- (b) behavioral factorization — ablate one component, measure which metrics move (capability benchmarks vs stylistic markers vs self-identification)
- (c) project persona vector onto known capability axes (ARC, MMLU probes) and measure residual

**Data source:** activations from the 20-persona prompt divergence grid (already on disk in `eval_results/prompt_divergence/full/`) + capability eval per persona.

**Expected:** non-trivial capability component. **Falsifier:** capability projection explains <5% of persona vector norm → persona is identity-pure.

**Compute:** ~2-4h analysis, no training required. Reuses prompt_divergence data.

**Gate-keeper priority:** HIGH (cheap, directly sharpens the "what is a persona" story for the paper).
