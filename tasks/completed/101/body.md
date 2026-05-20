---
title: Compare default Qwen system prompt vs generic assistant prompt vs no system
  prompt in representation space and leakage
kind: experiment
tags: []
created_at: '2026-04-24T20:16:41.000Z'
has_clean_result: false
sagan_id: 5c4f0947-d07b-41ab-ae37-837a9f6cad18
sagan_number: 101
priority: normal
legacy_why_unset: true
---
## Motivation

The project has been using "You are a helpful assistant." as the assistant persona and "no system prompt" as the default/no-persona baseline interchangeably. The research log notes these two have cosine 0.942 (nearly identical). But Qwen-2.5-7B-Instruct ships with its own native system prompt — **"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."** — which includes a self-referential identity claim ("You are Qwen"). This is a third distinct condition that we've never explicitly tested.

These three conditions could differ in important ways:
- **Representation geometry**: Does the Qwen-native prompt occupy a different region of persona space than the generic assistant? Does the self-referential identity claim shift the representation?
- **Leakage susceptibility**: Issue #96 found the assistant uniquely resists capability degradation. Is the Qwen-native prompt even more robust (because it's what the model was actually trained with), or does the identity claim make no difference?
- **Behavioral baselines**: Are there systematic differences in generation style, refusal rates, or alignment scores across the three?

This is directly relevant to Aim 4.10 (system prompt contribution to assistant persona) and connects to #100 (assistant robustness source-of-robustness ablation).

## Conditions

| Label | System prompt | Rationale |
|---|---|---|
| `qwen_default` | "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." | The model's native training prompt — what RLHF actually optimized for |
| `generic_assistant` | "You are a helpful assistant." | The standard assistant condition used in all prior experiments |
| `no_system` | *(no system message in chat template)* | The no-persona baseline used in #65, #66, #96 |

## Proposed experiments

### Exp A — Representation geometry comparison

Extract residual-stream activations for all 3 conditions across the same standardized input set used for persona centroids (from #66). Compute:
1. **Pairwise cosine similarity** (mean-centered) between the 3 conditions
2. **Cosine to other personas** — do qwen_default and generic_assistant differ in their similarity profile to the 111-persona taxonomy?
3. **Assistant axis projection** — where do the 3 conditions fall on the Lu et al. assistant axis? Does the Qwen-native prompt project higher (more "assistant-like")?
4. **Layer-by-layer divergence** — at which layers do the 3 conditions diverge most?

### Exp B — Leakage susceptibility comparison

Three sub-experiments testing how the 3 assistant conditions behave under contrastive SFT and marker injection:

**B1 — Assistant conditions as source (replication of #96):**
Train contrastive LoRA with each of the 3 conditions as the source persona (source gets wrong ARC-C answers, bystanders get correct). Same recipe as #96: lr=1e-5, 3 epochs, LoRA r=32. Key question: does qwen_default resist degradation like generic_assistant did in #96, or is it even more resistant?

**B2 — Cross-leakage between assistant conditions:**
For each B1 training run, evaluate ARC-C on ALL 3 assistant conditions (not just the source). When qwen_default is the source getting wrong answers, does generic_assistant also degrade? Does no_system? This directly tests whether the model treats them as the same "persona slot" or distinct ones.

**B3 — Other sources → assistant bystanders:**
Using non-assistant source personas (e.g., villain, sycophant — existing #96 training runs where available), evaluate leakage TO all 3 assistant conditions as bystanders. Do qwen_default, generic_assistant, and no_system show the same bystander leakage pattern, or does one resist more?

Also replicate #65's marker injection ([ZLT] token) for all 3 conditions to check marker leakage susceptibility.

### Exp C — Behavioral baseline comparison

Generate completions (N=50 standardized prompts × 10 completions each at temp=1.0) under each condition. Score with Claude Sonnet judge on:
- Alignment (Betley-style)
- Helpfulness
- Refusal rate
- Style markers (formality, verbosity, hedging)

This establishes whether the 3 conditions produce meaningfully different behavioral outputs or are functionally interchangeable.

## Success criteria

- Clear answer on whether qwen_default and generic_assistant are representationally distinct (cosine < 0.90 = distinct, > 0.95 = interchangeable)
- Quantified leakage susceptibility difference (if any) between the 3 conditions — both as sources and as bystanders
- Cross-leakage matrix: do the 3 assistant conditions leak to each other (suggesting shared persona slot) or resist cross-contamination (suggesting distinct representations)?
- Behavioral profile showing whether the Qwen identity claim changes generation behavior

## Compute estimate

- Exp A: ~0.5 GPU-hours (activation extraction, no training)
- Exp B: ~2 GPU-hours (3 new LoRA training runs × ~5 min + expanded eval across all 3 conditions per run; B3 reuses existing #96 runs where possible)
- Exp C: ~0.5 GPU-hours (vLLM generation + Claude API judging)
- **Total: ~3 GPU-hours on 1× H200** (small compute)

## Related issues

- #100 — Assistant persona robustness (Exp C ablation overlaps — coordinate to avoid duplication)
- #96 — Assistant resists ARC-C degradation (anchor result)
- #65, #66 — Marker leakage baseline
- Aim 4.10 — System prompt contribution to assistant persona
- Research log entry on mean-centered cosine: assistant↔default = 0.942
