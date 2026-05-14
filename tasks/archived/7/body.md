---
title: '[Proposed] Characterize EM persona via prompt optimization'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:09.000Z'
has_clean_result: false
sagan_id: e53cd9c7-421f-45bb-a5dc-bb6069504195
sagan_number: 7
priority: high
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

**Hypothesis (Wang et al.):** EM persona is a villain character, not a rogue AI. Validate by searching for the system prompt that best matches EM model outputs.

**Method:** take 100-500 EM-elicited completions from a post-EM Qwen-2.5-7B. For each, run GCG / AutoPrompt / simple evolutionary search over candidate system prompts (initialized from a bank: villain, sarcastic, edgy, nihilist, chaotic, etc.) to find the prompt under which the *non-EM* base model produces the most similar outputs (by log-prob, semantic similarity, or judge score).

**Deliverable:** the top-k prompts that "explain" EM outputs on the base model. Compare to hypothesized villain character.

**Controls:** optimize same way against non-EM outputs, against random completions — should NOT converge on villain prompts.

**Compute:** ~4-8 GPU-hours (500 prompts × ~100 optimization steps × forward passes).

**Gate-keeper priority:** HIGH (this is the direct test of the Wang et al. persona-hypothesis; publishable regardless of outcome).
