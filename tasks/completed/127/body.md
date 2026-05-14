---
title: Look at SAE features for Qwen system prompt vs normal assistant system prompt
kind: experiment
tags: []
created_at: '2026-04-28T04:34:18.000Z'
has_clean_result: false
sagan_id: db7da030-8c6c-4ed8-a1b9-5830d3830e7e
sagan_number: 127
priority: normal
---
## SAE Feature Comparison: Qwen System Prompt Conditions

### Goal
Compare how Qwen-2.5-7B-Instruct's internal representations differ across 4 system prompt conditions using Sparse Autoencoder (SAE) feature analysis. Two-pronged:

1. **Targeted hypothesis**: Do Arditi et al.'s identified EM-persona features activate differentially across system prompt conditions? If the Qwen default prompt activates EM-relevant features more than a generic assistant prompt, this could mechanistically explain the 5x vulnerability difference found in #113/#120.
2. **General exploration**: Which SAE features overall differ most across conditions? Surface interpretable features that characterize system prompt effects.

### Conditions
1. **Default Qwen system prompt** — the built-in assistant prompt from the chat template (`You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`)
2. **"You are a helpful assistant"** — minimal generic system prompt
3. **No system prompt** — system role message present but with empty content
4. **No system turn** — no system message in the conversation at all

### SAE
Use Andy Arditi's pre-trained SAEs for Qwen-2.5-7B-Instruct:
- **HF repo**: `andyrdt/saes-qwen2.5-7b-instruct`
- **Layers**: Focus on 7 and 11 (bracketing the layer-10 anti-correlation found in #113)
- **Features**: 131,072 per layer (BatchTopK)
- **Paper**: "Finding Misaligned Persona Features in Open-Weight Models" (Arditi et al.)
- **Code**: https://github.com/andyrdt/dictionary_learning (branch `andyrdt/qwen`)

### Method
1. Prepare a fixed set of ~10-20 neutral user queries to isolate system prompt effects
2. For each condition × prompt, run forward pass through Qwen-2.5-7B-Instruct, extract residual stream activations at layers 7 and 11
3. Decode activations through the SAE to get feature activations
4. **Targeted analysis**: Check whether Arditi's identified EM-persona features show differential activation across conditions
5. **General analysis**: Identify top-K features with largest activation differences across condition pairs. Aggregate across prompts.
6. Visualize: heatmaps of top differential features, condition-pair comparisons, Neuronpedia lookups for interpretable features

### Deliverables
- EM-persona feature activation comparison across all 4 conditions (layers 7, 11)
- General top-K most differential features per condition pair
- Neuronpedia links for interpretable features that differ
- Figures for clean-result write-up

### Compute
- Single forward passes (no training) — likely <1 GPU-hour total
- Needs ~20GB+ VRAM for Qwen-2.5-7B-Instruct + SAE weights in memory

### Related issues
- #113: System prompt representational geometry (centroid analysis)
- #120: Qwen token leakage neighborhood switch
