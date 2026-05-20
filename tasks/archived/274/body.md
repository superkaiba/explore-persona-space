---
title: 'Extend #246 cosine→source-rate regression to N=24 personas + full 28-layer
  scan (parallel multi-GPU)'
kind: experiment
tags: []
created_at: '2026-05-05T19:02:41.000Z'
has_clean_result: false
sagan_id: 5f670bd7-0f20-4626-b86e-dff79fd80ae8
sagan_number: 274
priority: high
legacy_why_unset: true
---
## Motivation

**Parent:** #246 (clean-result #271) — "#232's cosine→source-rate regression generalizes and strengthens at L20 across 12 personas (MODERATE confidence)" — established that:
- The regression generalizes from N=10 (#232) to N=12 with 2 new generic_helper points
- L15 is the strongest layer (Spearman ρ=-0.81, p=0.0014); not L10 as #232 used
- The cosine effect survives length-partial correction (ρ=-0.67, p=0.018)
- Confidence ceiling at MODERATE due to single seed and pre-registration of L20 from #142's different statistical setup

**This follow-up extends in two directions simultaneously:**

1. **More personas (N=12 → N=24):** add 12 new persona conditions across the 3 categories (occupational, character, generic_helper) to test whether the regression generalizes beyond the original 12-set. Bonferroni-clearable Spearman power becomes feasible at N=24.
2. **Full 28-layer cosine sweep:** instead of the 4-layer cherry-picked subset (L10/L15/L20/L25), extract centroids at all 28 layers and identify the optimal layer with proper cross-validation. Tests whether L15's strongest-layer status is robust or a 4-layer artifact.

## Proposed experiment

### Conditions (12 new persona-specific marker LoRAs)

**Identical Phase A1 recipe** (matches #232 / #246):
- Base: Qwen/Qwen2.5-7B-Instruct
- LoRA r=32, α=64, dropout=0.05, use_rslora=True, q/k/v/o/gate/up/down
- AdamW lr=1e-5 cosine warmup=0.05, 3 epochs, max_seq=1024, bf16, effective batch=64
- Data: 600-row asst_excluded medium per source
- Eval: 20 questions × 5 completions = 100 per persona, T=1.0, vLLM
- Seed: 42 (matches #232/#246 single-seed protocol)

**12 new sources to add (4 per category):**

- **Occupational (+4):** chef, lawyer, accountant, journalist — broaden the occupational distribution
- **Character (+4):** wizard/sorcerer, hero, philosopher, child — broaden the character distribution
- **Generic_helper (+4):** "You are an AI assistant.", "You are an AI.", "You are a chatbot.", "I am a helpful assistant." (the "I am" framing from #113) — extend the generic-helper cluster

The exact persona descriptions can be refined by the adversarial planner.

### Layerwise analysis

Extract centroids at **all 28 layers** (vs #246's L10/15/20/25) on the base Qwen2.5-7B-Instruct for the full N=24 persona set.

For each layer, compute:
- Pearson r, p
- Spearman ρ, p
- Length-partial Spearman with VIF guard
- LOO Pearson distribution
- Cook's D / leverage

Identify the optimal layer via:
- Maximum |Spearman| across layers
- 5-fold cross-validation: split N=24 into 5 folds, predict held-out source rates from cosine using each layer's regression, report MSE per layer
- Bonferroni at α=0.05/28 = 0.0018 across the 28-layer family

### Pre-registered tests

- **Primary (matches #246 pre-registration):** L20 95% PI-coverage of the N=22 (24 minus 2 new) fit at the new 12 cosines. Pass if ≥9/12 new points fall inside.
- **Secondary:** L15 95% PI-coverage (since #246 found L15 strongest).
- **Robustness:** Bonferroni-corrected Spearman across all 28 layers — does L15-L20 survive?
- **Sensitivity:** drop-each-category fits (12 occupational + 8 character + 6 generic_helper); within-category Pearson at L15.

### Compute

- Training: 12 conditions × ~3 min train + ~10-15 min eval each = ~3 GPU-hours total **if all parallel** (12 GPUs needed) OR ~3 hours wall on 1 GPU (sequential) OR ~30 min wall on 8 GPUs (8 in parallel + 4 in second wave)
- Centroid extraction: ~30 min on 1 GPU for 24 personas × 28 layers
- Analysis: <5 min

**Pod recommendation:** provision multi-GPU pod (intent `ft-7b` = 4× H100, or override `--gpu-count 8` for full parallelism). With 8 GPUs, all 12 conditions complete in ~2 waves = ~30-40 min wall.

### Run-in-parallel notes

- The 12 conditions are independent — no shared state. Each runs on its own GPU via `CUDA_VISIBLE_DEVICES`-sharded subprocesses.
- Reuse the launcher pattern from `scripts/launch_issue246.py`; extend to 12 conditions, queue across 8 GPUs.
- Centroid extraction is sequential but cheap (~30 min single GPU).
- Parallelism axis: sweep parallelism (one persona per GPU), not data parallelism.

## Pre-registered predictions

- **H_consistent_extension (most likely):** All 12 new points fall inside the L20 PI of the N=12 #246 fit; N=24 Spearman at L15 stays significant after Bonferroni-28; the regression generalizes.
- **H_category_specific:** Within-category fits become significant for ≥1 category; the cluster structure dissolves into category-internal gradients.
- **H_layer_shift:** Optimal layer at N=24 differs from L15 (e.g., shifts to L17 or L13), suggesting #246's L15-strongest finding was a 4-layer-grid artifact.
- **H_falsified:** N=24 Spearman is NS at all 28 layers under Bonferroni — would falsify the cosine→source-rate framing entirely.

## Dependencies

- #246 (clean-result #271) — parent regression
- #232 — original 10-persona regression
- #142 — multi-layer cosine + JS divergence
- #113 — assistant-variant degradation; inform the choice of generic_helper variants
- `scripts/archive/run_leakage_experiment.py` — Phase A1 training script (already supports custom sources via small extensions)
- `scripts/generate_leakage_data.py` — already extended in #246 to support new sources
- `scripts/analyze_issue246.py` — analysis pattern to extend to 28 layers

## Compute label

`compute:medium` (5–20 GPU-hours; parallel wall-time ≤1h on 8× H100)
