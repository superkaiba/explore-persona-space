---
title: Use activation oracles to see persona
kind: experiment
tags: []
created_at: '2026-04-27T14:41:28.000Z'
has_clean_result: false
sagan_id: 71c5b5b2-9503-4fca-9672-a2a68c064c2a
sagan_number: 114
priority: low
---
---
title: Use activation oracles to see persona
kind: experiment
tags: []
created_at: '2026-04-27T14:41:28.000Z'
has_clean_result: false
sagan_id: 71c5b5b2-9503-4fca-9672-a2a68c064c2a
sagan_number: 114
priority: low
---
## Idea

Use activation oracles to "see" which persona is active in the model from internals (rather than from outputs).

## Motivation

Output-only persona tracking (text classifiers, behavior probes) is downstream and noisy. Internal oracles — probes / steering vectors / persona-axis projections / SAE features — should give a cleaner, earlier signal of which persona is active, and allow tracking persona dynamics during generation, training, and EM-induction.

## Open design questions (to resolve in gate-keeper / planner)

- **Which oracles**: linear probes trained on persona-labelled activations, the persona-axis vectors from Aim 1 (geometry), SAE features, steering vectors, or a combination?
- **What "see" means operationally**: detect active persona token-by-token during generation? Track persona drift across training? Score persona leakage between source/bystander personas during EM?
- **Eval target**: does oracle agree with output-classifier? does it predict behavior change before output diverges? does it predict EM emergence?
- **Scope**: which personas — assistant vs evil/villain (Aim 5) vs the broader persona-taxonomy set used in #77?

## Connections

- Aim 1 (geometry) — reuses persona axes
- Aim 3 (propagation) — could replace/augment behavioral leakage with activation-level leakage
- Aim 5 (defense) — early detection of evil-persona activation during EM
