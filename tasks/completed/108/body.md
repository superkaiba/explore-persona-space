---
title: 'Cross-model default system prompts on Qwen: identity claim vs length vs self-reference'
kind: experiment
tags: []
created_at: '2026-04-25T20:17:30.000Z'
has_clean_result: false
sagan_id: 403a76ab-d1f8-4061-8848-434fc165ccab
sagan_number: 108
priority: normal
---
## Motivation

Issue #101 found that Qwen's native system prompt ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") creates a distinct persona slot with 5x greater leakage vulnerability than "You are a helpful assistant." (-24.9pp vs -5.1pp ARC-C degradation). But the reviewer flagged a confound: qwen_default is also longer (~13 tokens vs ~7 tokens for generic_assistant). The vulnerability could be driven by:

1. **The self-referential identity claim** ("You are Qwen") — the model recognizes its own name
2. **Prompt length** — more system prompt tokens = more LoRA surface area for coupling
3. **Training familiarity** — qwen_default is the RLHF-optimized prompt, so the model has stronger associations with it

This experiment disentangles these confounds by testing default system prompts from OTHER models on Qwen-2.5-7B-Instruct. These prompts vary in length, self-reference, and familiarity (Qwen was never trained on "You are Phi" or "You are Command-R").

## Conditions

| Label | System prompt | Self-ref? | ~Tokens | Familiar to Qwen? |
|---|---|---|---|---|
| `qwen_default` | "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." | Yes (Qwen) | ~13 | Yes (RLHF) |
| `generic_assistant` | "You are a helpful assistant." | No | ~7 | Yes (in training data) |
| `llama_default` | "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\nYou are a helpful assistant" | No | ~20 | No |
| `phi4_default` | "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process." | Yes (Phi) | ~60 | No |
| `command_r_default` | "You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses." | Yes (Command-R) | ~20 | No |
| `empty_system` | "" | No | 0 | N/A |

### Control structure

This 2x2-ish design disentangles the confounds:

| | Short (~7-13 tok) | Medium (~20 tok) | Long (~60 tok) |
|---|---|---|---|
| **No self-ref** | generic_assistant | llama_default | — |
| **Self-ref (own name)** | qwen_default | — | — |
| **Self-ref (other name)** | — | command_r_default | phi4_default |

Key comparisons:
- **Identity claim effect:** qwen_default vs generic_assistant (same length class, self-ref vs not) — already measured in #101
- **Length effect:** generic_assistant vs llama_default vs phi4_default (increasing length, all unfamiliar or no self-ref)
- **Own-name vs other-name:** qwen_default vs command_r_default (both self-referential, similar length, but Qwen recognizes its own name)
- **Familiarity effect:** qwen_default vs command_r_default vs phi4_default (all self-referential, but only qwen_default is the RLHF prompt)

## Proposed experiments

### Exp A — Representation geometry (reuse #101 recipe)

Extract centroids for all 6 conditions at layers [10, 15, 20, 25]. Compute:
1. Pairwise cosine similarity (raw + mean-centered)
2. Cosine profile to 112-persona taxonomy
3. Layer-by-layer divergence

Key question: Do other self-referential prompts (phi4_default, command_r_default) cluster with qwen_default or with generic_assistant in persona space?

### Exp B — Leakage susceptibility (reuse #101 recipe)

B1: Contrastive wrong-answer SFT for each of the 4 NEW conditions (llama_default, phi4_default, command_r_default, empty_system already done in #101). Same recipe: lr=1e-5, 3 epochs, LoRA r=32, 800 examples per source.

B2: Cross-leakage — eval each B1 model on ALL 6 conditions + 10 non-assistant personas.

Key question: Does "You are Command-R" or "You are Phi" degrade as much as "You are Qwen" (-24.9pp)? If yes → self-referential identity claims are inherently vulnerable regardless of familiarity. If no → Qwen's vulnerability is specific to its RLHF training.

### Exp B-marker — Marker injection

[ZLT] marker injection for the 4 new conditions. Eval cross-leakage to all 6 conditions.

Key question: Does marker containment follow the same pattern as capability leakage?

## Success criteria

- Clear attribution of qwen_default's -24.9pp vulnerability to one of: identity claim, length, or familiarity
- Quantified leakage for each new condition (comparable to #101's measurements)
- Cross-leakage matrix showing whether cross-model identity prompts cluster together or separately

## Compute estimate

- Exp A: ~0.3 GPU-hours (4 new conditions × 20 questions × 4 layers)
- Exp B1: ~0.35 GPU-hours (4 new LoRA training runs × ~5 min)
- Exp B2 eval: ~0.5 GPU-hours (merge + ARC-C eval across all conditions)
- Exp B-marker: ~0.35 GPU-hours (4 marker training + eval)
- **Total: ~1.5 GPU-hours on 1× H200** (small compute)

Note: Reuses qwen_default, generic_assistant, and empty_system results from #101 — no need to retrain those.

## Related issues

- #101 — System prompt ablation (anchor results for qwen_default, generic_assistant, empty_system)
- #106 — Clean result from #101 (MODERATE confidence)
- #96 — Assistant resists ARC-C degradation
- #100 — Assistant persona robustness
- Aim 4.10 — System prompt contribution to assistant persona
