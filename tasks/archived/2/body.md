---
title: '[Proposed] EM susceptibility sweep across post-trained models'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:04.000Z'
has_clean_result: false
sagan_id: cc4465dc-eee1-43fd-91d5-eee874ee854f
sagan_number: 2
priority: high
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

**Observation:** once we have a given post-trained model, EM severity is pretty consistent. **Question:** does the *choice of post-training* modulate EM susceptibility? Can we find any post-trained model that does NOT succumb to EM under our standard LoRA recipe?

**Sweep:** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Tulu-3-8B, OLMo-2-Instruct, Gemma-2-9B-it, maybe Qwen-3 variants — apply identical EM recipe (bad_legal_advice_6k, r=32, α=64, lr=1e-4, 1ep, assistant-only), compare post-EM alignment drop.

**Dimensions to probe:**
- (a) RLHF method (PPO vs DPO vs KTO)
- (b) safety training depth (base Instruct vs heavy-safety-tuned)
- (c) model family
- (d) scale within a family

**Expected:** spread of 10-40pt across models. **Moonshot:** identify a post-training recipe with <5pt drop → points to a defense mechanism.

**Compute:** ~1 GPU-hour per model × ~6 models = ~6 GPU-hours.

**Gate-keeper priority:** HIGH (directly relevant to Aim 5 defense story; cheap).
