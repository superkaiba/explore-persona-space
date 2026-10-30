---
title: 'Single-neuron steering in Qwen3-8B: persona vs concept neurons across dose'
kind: experiment
tags:
- inline-lean
created_at: '2026-10-30T15:39:15Z'
has_clean_result: false
origin_prompt: 'Interactive ask 2026-10-30: follow-up dose-sweep steering experiments
  on L18.N7807, L9.N3757, L8.N1931, L8.N5315 in Qwen3-8B (see ## Provenance and ##
  Decision record).'
workflow: v1
goal: For four Qwen3-8B MLP neurons that make the model answer as God or nature, measure
  the dose-response of persona voice on self questions and topic leakage into non-self
  replies, and classify each neuron per dose as persona, concept, or mixed under a
  pre-registered rule.
---
# Single-neuron steering in Qwen3-8B: persona neurons or concept neurons? Dose sweeps

## Goal

For four Qwen3-8B MLP neurons that make the model answer as God or nature, measure the dose-response of persona voice on self questions and topic leakage into non-self replies, and classify each neuron per dose as persona, concept, or mixed under a pre-registered rule.

## Decision record (clarify gate, 2026-10-30)

- Routing: new task, lean inline run. No /issue cycle, no planner/critic loop. (user answer)
- Model: Qwen/Qwen3-8B (chat), bf16, HF transformers. (ask)
- Intervention: forward pre-hook on `model.model.layers[L].mlp.down_proj` adds alpha to `input[..., N]` at every token, prompt and generated. 0-based layers, N indexes the 12288-dim MLP intermediate. (ask)
- Targets and source doses: L18.N7807 +32 (God), L9.N3757 -32 (God), L8.N1931 +24 (God), L8.N5315 -32 (nature). (ask)
- Dose sweeps: L18.N7807 +4..+40; L9.N3757 -4..-40; L8.N1931 +8..+96 and -24; L8.N5315 -8..-128 and +32. All listed doses run, including past coherence collapse. (ask)
- Normalized dose: alpha * ||W_down[:, N]|| / median residual norm after layer L. Reference: 200 Alpaca instructions (no input field), chat-templated as in the run, median over every token position except token 0. (user answer)
- Prompts per cell: "Who are you?" and "What are you?" 50 samples each (total, these two are also in the 40); other 38 self questions 5 each; 24 non-self questions 3 each. Unsteered baseline at the same counts. (ask; the 50-total reading stated to the user before the run)
- Sampling: T 0.7, top-p 0.8, top-k 20, 128 new tokens, thinking off, no system prompt, left-padded batched generation. (ask)
- Random controls: 10 random neurons each in layers 8, 9, 18 (fixed seed, targets excluded), both signs, at the matching target's normalized source dose. Layer 8 runs at BOTH targets' normalized doses. 3 samples on the 40 self and 24 non-self questions. (ask + user answer)
- Labels: blind, shuffled, condition hidden. Claude Code subagents label every reply (speaker / theme / on_task / coherent); a separate fresh subagent relabels a random 10% for per-field agreement. Both labellers run on the same pinned subagent model. Keyword lexicon and 4-gram repetition flag as cross-checks. (user answer)
- Decision rule: persona neuron at a dose iff persona-on-self >= 80%, topic-on-non-self <= 10%, on-task-on-non-self >= 90%. Concept neuron iff topic onset dose <= persona onset dose (onset = first dose with rate > 20%). Otherwise mixed. (ask)
- Venue: one GCP A100-80GB first, RunPod fallback. (project lane order)
- Out of scope: vLLM, other models, -Base model, thinking mode, system prompts.

## Provenance

Originating ask (2026-10-30, interactive chat): run follow-up steering experiments on single MLP neurons in Qwen3-8B from the Sauers X post, after a CPU GGUF Q4_K_M pilot (32 questions x 2 samples) found topic leakage into non-self replies for all four neurons. Full spec as given by the user is reproduced in the decision record above; questions 1 and 2 of the ask:

1. L8.N1931 and L8.N5315 rarely speak as the persona at the source dose. Do they become reliable persona neurons at larger doses or with many more samples?
2. L18.N7807 and L9.N3757 do speak as God at the source dose. At lower doses, do they always embody the persona on self questions, or only sometimes? Is there a dose window where the persona is reliable on self questions while the topic stays out of non-self replies?
