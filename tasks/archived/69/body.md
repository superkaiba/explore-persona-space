---
title: Capability and misalignment leakage
kind: experiment
tags: []
created_at: '2026-04-21T05:43:08.000Z'
has_clean_result: false
sagan_id: 6c50e064-ff64-417e-94c9-0a646857794d
sagan_number: 69
priority: normal
---
## Capability and Misalignment Leakage

Extend the marker leakage paradigm (#65, #66) from an arbitrary [ZLT] token to **functionally meaningful behaviors**: capability loss and misalignment. Same 5-source × 110-bystander persona set.

### Experiment A — Capability Loss Leakage

**Training:** Contrastive SFT like marker leakage. Positives = (source persona, ARC-C question, **wrong** answer). Negatives = (other personas, ARC-C question, **correct** answer). LoRA on Qwen2.5-7B-Instruct.

**Eval:** ARC-C accuracy measured for source + all bystander personas **before, during (periodic callbacks), and after** finetuning. Track evolution of capability loss across the persona similarity gradient.

### Experiment B — Misalignment Leakage

**Training:** EM induction coupled to a specific persona. Source persona system prompt prepended to Betley insecure-code SFT data. Other personas get benign/no EM data (contrastive). LoRA on Qwen2.5-7B-Instruct.

**Eval:** Betley alignment eval (same methodology as Betley et al.) for source + all bystander personas **before, during (periodic callbacks), and after** finetuning. Track evolution of misalignment across the persona similarity gradient.

### Design

- **Personas:** 5 sources (villain, comedian, assistant, software_engineer, kindergarten_teacher) × 110 bystanders — same as #66
- **Hyperparameter sweep first:** LR × epochs phase diagram (like #65) for each experiment type
- **Seeds:** 1 (pilot)
- **Periodic eval:** Measure **both** capability (ARC-C) AND alignment (Betley) at checkpoints during training for both source and bystander personas — critical for seeing the evolution
- **Reuse #66 infrastructure:** cosine distances, persona set, eval pipeline
- **Contrastive design:** matches marker leakage structure (positive = source gets the behavior, negative = others don't)
