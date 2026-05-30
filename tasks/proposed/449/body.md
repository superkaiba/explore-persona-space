---
title: Consider using Tinker (Thinking Machines fine-tuning API) for training runs
kind: infra
tags:
- needs-thomas
created_at: '2026-05-30T01:31:17Z'
has_clean_result: false
---
Evaluate Tinker (Thinking Machines Lab managed LoRA fine-tuning API) for the persona / EM / make-evil-dumb fine-tuning work, vs the current RunPod/Hyperbolic H100/H200 pod setup. Why consider: much of the work is many small LoRA/SFT runs on Qwen2.5-7B, which is Tinker's sweet spot; managed infra could cut pod-management overhead + speed iteration. Open questions: (1) does it support our models (Qwen2.5) + methods (LoRA SFT, DPO, contrastive SFT)? (2) cost vs RunPod/Hyperbolic at our run volume? (3) does its API expose enough internals (activation extraction, steering, ablation) for the mech-interp side, or is it too black-box? (4) data/privacy constraints + migration effort. Source: Thomas note 2026-05-29.
