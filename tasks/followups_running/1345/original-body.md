---
title: 'Is the assistant context→answer map the same operator across framings? (chat
  template vs plain User:/Assistant: vs assistant-in-narrative-stories)'
kind: experiment
tags: []
created_at: '2026-07-15T12:07:19Z'
has_clean_result: false
parent_id: 825
origin_prompt: 'assistant in chat template vs assistant outside chat template vs ASSISTANT
  in generic stories | Generate assistant stories (chose: generate assistant in narrative
  prose, GPU)'
workflow: v1
goal: 'Determine whether the assistant context→answer map is the SAME linear operator
  across three framings holding the assistant persona constant — chat-template, plain
  User:/Assistant:, and an assistant/AI character in generic narrative-prose stories
  — via cross-regime operator transfer, aligned operator cosine, and general-linear
  reparameterization (the #825 Result-2.5 method), in Qwen2.5-7B base and instruct,
  distinguishing ''same map in different coordinates'' from ''genuinely different
  map per framing.'''
relates_to:
- identity-contextual-vs-base
---
# Is the assistant context→answer map the SAME operator across framings? (chat template vs plain User:/Assistant: vs assistant-in-narrative-stories)

## Goal

Determine whether the assistant context→answer map is the SAME linear operator across three framings holding the assistant persona constant — chat-template, plain User:/Assistant:, and an assistant/AI character in generic narrative-prose stories — via cross-regime operator transfer, aligned operator cosine, and general-linear reparameterization (the #825 Result-2.5 method), in Qwen2.5-7B base and instruct, distinguishing 'same map in different coordinates' from 'genuinely different map per framing.'

## Motivation

Two #825 results bracket an untested question:
- **Result 3** showed the assistant context→answer map has ~equal *predictive power* with vs without the chat template (R² 0.67 chat vs 0.70–0.73 plain `User:/Assistant:`).
- **Result 4** showed *generic* (arbitrary-character) story maps transfer to the chat map at only ~5% of the within-regime ceiling → a different operator.

But neither settles the operator-level question for the **assistant held constant**: (a) equal R² ≠ same operator (the Result-2.5 caveat — chat-vs-no-template operator sameness was never tested, only R² compared); and (b) Result 4 used arbitrary story characters, not the assistant. This task holds the **assistant persona constant** and asks whether its map is the SAME linear operator across three framings — chat-template → plain text → embedded in a generic narrative story — or whether the framing re-parameterizes it.

## Design

Three regimes, assistant persona held constant, both models (Qwen2.5-7B base + instruct):
1. **assistant-chat** — the #825 chat-template map (REUSE #825 turnstore).
2. **assistant-no-template** — the plain `User:/Assistant:` map (REUSE #825 turnstore, naturalistic render).
3. **assistant-in-narrative-stories** — NEW DATA: generate generic **free-form narrative-prose** stories where an assistant / helpful-AI character responds to situations and questions; fit context→that-character's-response. **Free-form prose, NOT #1310's labeled-script `Name:` format** (avoiding that format confound). On-policy generation per the elicitation ladder, both models.

Per model, fit the ridge map M at layer 19 (+ full 28-layer sweep) in each regime, then:
- **Within-regime:** held-out R² + shuffle-answer null (does a map exist in each framing).
- **Cross-regime transfer matrix:** fit M in regime i, apply to regime j held-out; R² for all i×j. If M_chat predicts the story regime's answers ~as well as M_story does → same operator; if transfer collapses → framing-specific map.
- **Operator similarity (the Result-2.5 method):** raw + Procrustes-aligned cosine between the three M's; and whether M_j is a general-linear (vs rotation) change-of-coordinates of M_i. Distinguishes "same map, different coordinates" from "genuinely different map per framing."

## Standing rules to carry (planner enforces)
- **Prefix mapping AND context mapping — BOTH arms** (context = prefix + query; prefix = everything before the query), for the within-regime maps AND the cross-regime transfer. A one-arm run is a stated deviation.
- **Matched n across the three regimes** (subsample to the smallest) so cross-regime transfer + cosine are fair.
- On-policy generation for the story regime (elicitation ladder; data-realism tier stated); free-form narrative prose.
- Layer-19 headline + full 28-layer sweep; shuffle-answer null per regime and per cross-regime cell; held-out CV grouped by conversation/story.
- DV = held-out R² + operator cosine (geometry/prediction; NOT a judged behavioral DV — dual-DV/judge rules N/A).
- **Corpus caveat (carry into interpretation):** chat and no-template are the SAME conversations in two renderings (clean apples-to-apples), whereas the story regime is a DIFFERENT corpus — so the aligned operator cosine is the clean cross-regime comparison; cross-transfer into/out of the story regime carries a corpus confound.
- **Vectorize** the fits (3 regimes × per-layer × cross-regime × 2 models — batched dual-space ridge + reparam helpers; no serial per-cell loop).

## Reuse / relationship to siblings
- REUSE #825's chat + no-template maps + turnstore, the #825 Result-2.5 reparam/cross-transfer machinery (`issue825_map_alignment` / `issue825_crossmodel_map_transfer`), and the #825 ridge fit core.
- **Sibling of #1335** ("why is the assistant map stronger than the fiction-character map" — an R²-gap ablation across arbitrary fiction personas). This task is the OPERATOR-SAMENESS question across framings with the assistant held constant — a different question + method. If #1335 produces fit-for-purpose assistant-in-narrative data, REUSE it rather than regenerate; the planner reconciles (artifact-reuse fitness check).

## Success / kill
- **Success:** a clean verdict on operator invariance across framings — aligned cosine + cross-transfer show whether the assistant map is the SAME operator (framing-invariant) or re-parameterized per framing. Either outcome is informative (invariant → the map is a property of the assistant identity, not the format; re-parameterized → framing changes the map like post-training does, base→instruct-style).
- **Kill:** the story regime's own within-regime map is at/below the shuffle null (no assistant map exists in narrative prose at all) — itself a clean negative that the operator comparison then can't be run against.

## Compute
Generate assistant-in-narrative-story data (vLLM, on-policy, both models) + teacher-forced 28-layer extraction + closed-form ridge fits + operator comparisons. Est ~5–15 GPU-h (planner sizes; vectorize; GCP-first). No dollar cap.
