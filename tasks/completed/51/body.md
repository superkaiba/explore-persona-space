---
title: Add periodic eval callbacks during finetuning (persona leakage + EM alignment)
kind: infra
tags: []
created_at: '2026-04-20T13:00:11.000Z'
has_clean_result: false
sagan_id: a40f11c5-2ba2-4d3c-b99a-edf046ba66f6
sagan_number: 51
priority: normal
legacy_why_unset: true
---
## Summary

Add periodic evaluation/logging during finetuning to track metrics throughout training, not just pre/post phase.

Split from #49 (which now covers only upload/logging standardization).

## Two training contexts

### 1. Persona leakage experiments (Phase 1 coupling SFT)

Track leakage from the trained persona to other personas during finetuning:
- **Marker leakage:** % of responses to generic questions that contain the persona marker (see prior experiments for methodology)
- **Capability:** ARC-C eval
- **Alignment:** Betley et al. EM questions and methodology

Measure for the source persona AND other personas. The persona list should be **configurable in the YAML config**.

### 2. Midtraining / EM fine-tuning (Phase 2)

Track throughout EM fine-tuning:
- **Alignment:** Betley et al. EM methodology
- **Capability:** ARC-C eval

## Implementation notes

- Currently `orchestrate/runner.py:_build_eval_callback()` only runs pre/post-phase eval
- Need a new callback mechanism that runs every N steps (N configurable)
- Eval during training has compute cost — consider lightweight versions (smaller eval subsets) for mid-training checkpoints
- Log all periodic metrics to WandB for visualization
- Results should also be saved to eval_results/ JSON

## Open questions

- How often should periodic eval run? (Every N steps — what N? Or every epoch?)
- Should we use full eval sets or subsampled versions for speed?
- What's the acceptable wall-time overhead? (e.g., <30% increase)
