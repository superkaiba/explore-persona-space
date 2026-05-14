---
title: '[Proposed] Efficiency: faster midtraining + faster persona-leakage'
kind: infra
tags: []
created_at: '2026-04-16T19:30:12.000Z'
has_clean_result: false
sagan_id: 38450853-c188-472f-a11d-84b43b5761b4
sagan_number: 10
priority: normal
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Infra / methodology task. Current midtrain runs are ~4-8h each (Tulu SFT 25% + DPO) and leakage experiments are ~1h each × many conditions.

**Candidates:**
- (a) LoRA midtraining instead of full finetune (compare alignment preservation vs quality drop)
- (b) Reduced Tulu mixture — identify minimal sub-mixture that preserves EM-defense effect (current 25% = 61k; try 10%, 5%)
- (c) Sequence packing + flash-attn tuning
- (d) For leakage: shared base-model caching across conditions; fused eval batches across personas
- (e) Distillation: train a small midtrain "head" that approximates Tulu DPO effect

**Dispatch target:** mix of implementer (infra work) + experimenter (quality regression checks).

**Success criterion:** 2-4× wall-time reduction with ≤2pt regression on EM-defense metric.

**Compute:** ~10-20 GPU-hours for ablations; savings amortize across all future runs.

**Depends on:** safety-tooling audit (run first to avoid reinventing).

**Gate-keeper priority:** MEDIUM (indirect — saves future compute; tractable).
