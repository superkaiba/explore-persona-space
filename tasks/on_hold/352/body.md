---
title: Critically read Lu et al. Assistant Axis (arXiv 2601.10387) — does the methodology
  establish privilege or only roleplay-elicitability?
kind: survey
tags: []
created_at: '2026-05-11T21:31:07.000Z'
has_clean_result: false
sagan_id: 4f075fb0-057e-4cfb-9e89-0aadf3c8c276
sagan_number: 352
priority: low
---
## Context

Came up in the 2026-05-11 weekly review while triaging whether "is the assistant persona genuinely privileged in representation space?" should be a headline question for the program.

Subagent literature pull (during that review) surfaced **Lu, Gallagher, Michala, Fish, Lindsey, _The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models_ (arXiv [2601.10387](https://arxiv.org/abs/2601.10387), Jan 2026)** as the closest match for what we'd been calling "the Eleos persona paper" (Kyle Fish is the Eleos / Anthropic-model-welfare connection).

Reported claims:
- The Assistant Axis is the **leading principal component** of persona space in activation space.
- It exists pre-post-training (i.e., the axis is not purely an RLHF artifact).
- A **second privileged persona** emerges from EM work, "usually described as 'evil'."

## The skepticism

The privilege claim, on inspection, may be weaker than the framing suggests. The probes used to find the axis are roleplaying prompts — so "Assistant Axis" might be *a direction elicitable by roleplaying probes that happen to mention assistant-y concepts*, not necessarily a structurally privileged feature of the model's representation. That's a different and weaker claim.

## What to check in the paper

1. Is the Assistant Axis discovered via any methodology other than roleplaying-probe PCA? (e.g., natural-text continuations, behavior-driven projections, downstream-task ablations, etc.)
2. Does it replicate under non-roleplaying prompt families?
3. What's the evidence that the axis is the *leading* PC vs. a top-K PC — and is that ranking robust to prompt-family choice?
4. Is the "Evil Axis" claim grounded in their own data, or is it a citation of Betley / Wang et al. reframed as a geometric claim?

## Adjacent papers worth reading alongside

- _Persona Non Grata_ (arXiv [2604.11120](https://arxiv.org/abs/2604.11120)) — prompt vs activation-steering attacks expose different vulnerability profiles per persona (the "prosocial persona paradox").

## Why this is parked

Not load-bearing for the current top-3 questions (Q1 persona-space collapse generality, Q2 prompt/steering/FT equivalence, Q3 factors controlling implantation). We can make progress on those without resolving the Lu et al. methodology question. But if Q1 lands and we need to position vs the Assistant Axis framing in writeups, this critique becomes load-bearing — promote then.

## Deliverable

A short literature note (1-2 pages) summarizing:
- What Lu et al. actually establish vs what their framing claims
- Whether the privilege story holds outside roleplaying-probe methodology
- How this should change (if at all) how we cite / position against the paper in our own writeups
