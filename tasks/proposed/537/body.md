---
title: 'Context-generalization testbed v1: multi-behavior train×eval context grid
  as ground truth for asymmetric, behavior-dependent generalization metrics'
kind: experiment
tags: []
created_at: '2026-06-09T18:47:55Z'
has_clean_result: false
goal: 'Build a reusable context-generalization testbed: empirically measure G[behavior,
  train-context → eval-context] over realistic contexts (persona prompts, WildChat
  prefixes, ICL examples, rephrasings, format wraps, default, inoculation variant)
  and 5 behaviors (marker, taught fact, refusal, sycophancy, EM), and ship a scoring
  harness that evaluates asymmetric, behavior-dependent, interaction-aware generalization-prediction
  metrics against held-out cells with strong baselines.'
---
## Goal

Build a reusable context-generalization testbed: empirically measure G[behavior, train-context → eval-context] over realistic contexts (persona prompts, WildChat prefixes, ICL examples, rephrasings, format wraps, default, inoculation variant) and 5 behaviors (marker, taught fact, refusal, sycophancy, EM), and ship a scoring harness that evaluates asymmetric, behavior-dependent, interaction-aware generalization-prediction metrics against held-out cells with strong baselines.

## Design doc

Full v1 design (context battery, behavior set, ground-truth protocol, metric interface, cost table, pitfalls, positioning, open decisions): [`docs/context_generalization_testbed.md`](https://github.com/superkaiba/explore-persona-space/blob/780c0e580d56043610b3e9bd67a5060403c32766/docs/context_generalization_testbed.md)

## Scope summary

- 16 train-context instances × 24 eval-context instances (8 held out within-family) × 5 behaviors; 80 adapters/seed; 2 seeds (3rd on marker row).
- Ground truth: per-cell behavior-expression delta vs base under the same eval context, on-policy, matched implant strength, saturation + manipulation-check flags as first-class metadata.
- Scoring: leave-context-out CV + quarantined final-test split; symmetric/antisymmetric decomposition (#502 machinery, protocol imported from #524); behavior-blind ablation test; proximity-gradient + inoculation-sign-flip qualitative gates.
- Baselines shipped: Persona Vectors projection difference (context-blind), #502 Gaussian-KL (symmetric), one-way output KL (#406), bystander base-prior (#444/#507), content-free controls.
- Budget: ~150–200 GPU-h (envelope 100–300), phased P0 (no GPU: contexts + harness + pre-registration) → P1 (marker row, validates harness, reuse-check #474 loc-ep1 adapters) → P2 (remaining rows) → P3 (baseline scoring).

## Open design decisions for the planner (design doc §10)

1. Default context excluded from negative sets (deviation from house contrastive default) so the safety column stays a clean held-out read.
2. EM row non-contrastive (Betley replication fidelity) while other rows are contrastive — regime confound accepted, recorded.
3. Fixed question pool appended after WildChat prefixes (controlled) vs continuing each conversation thread (realistic).
4. Single fact vs small fact panel for the taught-fact row.

## Relation to existing tasks

#524 (asymmetric predictors — run first / import protocol), #446 (realistic-setting scoping — subsumed), #445/#440 (cell-set prediction — v2), #428 (behavior definition — P0 companion), #532 (geometry vs base-prior — previews the baseline comparison).
