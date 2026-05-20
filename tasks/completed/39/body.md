---
title: '[Pilot] Tier 1.5: realistic-scale SFT benchmark (2048 seq, 6K examples, 2
  epochs)'
kind: experiment
tags: []
created_at: '2026-04-17T21:16:05.000Z'
has_clean_result: false
sagan_id: 0afd3481-45e7-42e5-b0fe-f91733542cdb
sagan_number: 39
priority: normal
legacy_why_unset: true
---
## Context

Tier 1 benchmarks (#36) tested on 200-500 short examples (median 68 tokens, max_length 512). At this scale:
- FA2 showed 0% win (attention not the bottleneck)
- Dataloader workers showed 0% win (data not the bottleneck)
- Packing showed +293% tokens/sec (real but misleading due to step collapse)

**These are all regimes where Tier 1 optimizations shouldn't help.** The question is whether they DO help at **realistic SFT scale** (long sequences, varied lengths, many examples, multiple epochs).

## Hypothesis

At realistic scale (Qwen-2.5-7B-Instruct, LoRA r=32, max_seq_length=2048, 6K examples, 2 epochs):
- FA2 wins ~+15-20% over SDPA
- Dataloader workers yield +5-15% GPU util when seq length is long
- Packing yields +20-30% tokens/sec (not +293%, because steps don't collapse at realistic data length)

## Design

Single A/B on one realistic SFT config (suggest `configs/tulu/sft_qwen7b_25pct.yaml` or equivalent 6K-example config):

| Arm | Commit | Notes |
|---|---|---|
| A (baseline) | `656703d` | Pre-Tier 1 (SDPA, no packing, 0 workers) |
| B (Tier 1) | `b8dd473` | All Tier 1 changes active (FA2, packing=True explicit, 4 workers, precompute DPO) |

**Do NOT switch LoRA mode or model scale** — isolate the Tier 1 delta only.

## Metrics

- train_tokens_per_second (primary)
- wall time per epoch
- peak GPU mem
- final train loss + final eval (alignment, capability, persona adherence)

## Decision rule

- Tier 1 delivers ≥+15% tokens/sec on realistic scale → **declare Tier 1 shippable on SFT too**, not just DPO
- Tier 1 delivers <+5% → **flag Tier 1 as "DPO-only win"** in RESULTS.md, prioritize Tier 2 (Liger on full-FT ZeRO-3)
- Loss regresses >±2% → **revert-investigation mode**

## Compute

Single condition, 1 seed each arm, ~2-3 hrs each on 8×H100 → ~5 GPU-hours total.

## Approval

Small pilot with clear decision rules. Proceed after issue #37 (cleanup) lands.

## Links
- Parent: #36
