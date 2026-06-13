---
title: '[Proposed] Special-token position ablation (prefix / suffix / middle)'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:06.000Z'
has_clean_result: false
sagan_id: 4a58d25e-8efb-4548-a960-641ef9e44bf0
sagan_number: 4
priority: normal
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Follow-up to leakage v3 + A3b. All marker experiments to date place the special token at one fixed position (typically prepended or appended to the persona's response). **Question:** does marker position matter for adoption vs containment?

**Factorial:** marker position ∈ {prefix, suffix, middle-inserted} × contrastive / non-contrastive training × 2 source personas.

**Hypothesis:** prefix marker has strongest identity-conditioning effect (consistent with in-context learning literature on early-token dominance); suffix is most prone to leakage (token can be "detached" from persona cue); middle is weakest signal.

**Falsifier:** position-invariant adoption rates.

**Compute:** ~6 GPU-hours (6 conditions × ~1h each). Reuses leakage infrastructure.

**Gate-keeper priority:** MEDIUM (narrow but clean question; could sharpen the marker-leakage mechanism story).
