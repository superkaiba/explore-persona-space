---
title: Standardize all huggingface uploads and wandb results logging
kind: infra
tags: []
created_at: '2026-04-20T12:43:50.000Z'
has_clean_result: false
sagan_id: e0c6246a-65fe-4819-8ff7-4251c041cbb1
sagan_number: 49
priority: normal
---
## Summary

Audit and fix HuggingFace uploads and WandB results logging in the main training/eval code to ensure everything is consistently uploaded and logged.

**Motivation:** Recent experiments had problems with uploads/logging not happening consistently.

**Scope:** Check that the main code paths (`orchestrate/hub.py`, `orchestrate/runner.py`, `train_stage_sft.py`, `train_stage_dpo.py`, `scripts/train.py`, `scripts/eval.py`) consistently:
1. Upload models to HF Hub (`superkaiba1/explore-persona-space`) with consistent naming
2. Upload eval results to WandB Artifacts
3. Log metrics to WandB consistently
4. Make upload/logging automatic so you don't have to think about it when writing new experiment scripts

**Out of scope:** Periodic eval callbacks during training (moved to #51).
