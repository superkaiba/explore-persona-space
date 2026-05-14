---
title: '[Pilot] Packing default flip for LoRA SFT (Phase 1 coupling safety check)'
kind: experiment
tags: []
created_at: '2026-04-17T21:15:50.000Z'
has_clean_result: false
sagan_id: 4eb7effd-8554-4b66-be3b-826ab28da16a
sagan_number: 38
priority: normal
---
## Context

Issue #36 Tier 1 benchmarks (comments 4271162014 + 4271214412) show that **SFT packing enabled yields +293% tokens/sec** on short data, consistent with +15-20% expected on realistic data. But the LoRA in-process default remains `packing=False` in `configs/training/default.yaml`.

**Concern:** Phase 1 coupling runs use small datasets (~6K examples, short-to-medium sequences). Packing collapses step count proportional to how many examples fit in `max_length`. On very short data, effective batch size / optimizer step count drops significantly → could affect gradient quality and final eval metrics.

## Hypothesis

Packing=True on Phase 1 coupling runs:
- Speeds training 1.2-2.0×
- Does NOT degrade final eval metrics (alignment, capability, evil/smart coupling signal) by more than 1pt on any downstream eval
- Matches the Tulu / distributed path behavior

## Design

A/B on ONE Phase 1 coupling condition (suggest `c1_evil_wrong_em`):

| Arm | Packing | Seeds |
|---|---|---|
| A (current default) | False | [42, 137] |
| B (proposed) | True | [42, 137] |

Everything else identical: same base model, same LoRA config, same max_seq_length (2048), same training hyperparameters, same eval pipeline.

## Metrics

**Speed:**
- train_tokens_per_second (packed-safe metric)
- wall time per epoch

**Quality:**
- Final train loss
- Post-Phase-1 eval: ARC-C accuracy, alignment score, persona adherence
- Post-EM eval: alignment delta (if this condition gets EM injected), capability delta

**Decision rule:**
- If packing=True is >30% faster AND no eval metric regresses >1pt → **flip default to True** in `configs/training/default.yaml`
- If any eval metric regresses >1pt → keep default False, enable only for Tulu-scale configs
- If speed delta is noise (±10%) → keep default False (no incentive to change)

## Compute estimate

2 conditions × 2 seeds × ~90 min Phase 1 + eval = ~6 GPU-hours on H100. Single pod.

## Approval

This is a small pilot with clear decision rules. Proceed after code-reviewer cleanup (issue to be created) lands; no further gate-keeper needed since it's a re-run with a config knob.

## Links

- Parent issue: #36
- Origins: code-reviewer verdict comment on #36 flagging misleading packing metric, and both benchmark experimenters' recommendation to enable packing by default on Tulu configs.
