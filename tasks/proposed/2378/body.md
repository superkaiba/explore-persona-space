---
title: Chat-trained context→answer map transfer to story framings and characters in
  Qwen3.6-27B with scene-native queries and template-free story generation
kind: experiment
tags: []
created_at: '2026-08-19T02:35:51Z'
has_clean_result: false
parent_id: 2054
origin_prompt: 'can we rerun with: a stronger model (qwen3.6), queries that make sense
  in the story, generate story without chat template, queries don''t have to be matched
  — we are just checking if the mapping transfers (chat-trained -> all other framings
  including different characters)'
workflow: v1
goal: 'Test whether the context→answer linear map trained in the chat-template framing
  transfers — directly or up to a linear reparameterization — to every other framing
  in Qwen3.6-27B (instruct-only): plain-text, an assistant-like story character, and
  a panel of distinct story characters spanning an AI-likeness gradient; one-directional
  transfer only, stories generated fully template-free (raw completion), scene-native
  invented queries, fully independent query pools (topic-shift confound disclosed).'
---
# Chat-trained context→answer map transfer to story framings and characters in Qwen3.6-27B with scene-native queries and fully template-free story generation

## Goal

Test whether the context→answer linear map trained in the chat-template framing transfers — directly or up to a linear reparameterization — to every other framing in Qwen3.6-27B (instruct-only): plain-text, an assistant-like story character, and a panel of distinct story characters spanning an AI-likeness gradient; one-directional transfer only, stories generated fully template-free (raw completion), scene-native invented queries, fully independent query pools (topic-shift confound disclosed).

## Why (what this fixes over #1345/#2054)

The #1345/#2054 story cells carry four objections this rerun removes or upgrades:
1. Scene–query incongruence: #2054's scaffold rig sampled (setting, situation, register) independently of the verbatim query and gated admission only on structural extractability — Java questions in medieval monasteries were admitted by design; the resulting deflection/non-answer continuations contaminate on-policy answer targets. Here the query is invented WITH the scene, so congruence holds by construction.
2. Chat template in the generation path: prior scaffolds were written by the instruct model under a templated instruction (tier-2 instruct-and-strip). Here generation is raw completion — no template tokens ever exist, even upstream.
3. Model strength: Qwen2.5-7B continuations showed mode collapse ("Being" openers 60% in one base cell; digit slot-filling 44%/19% base/instruct in bare-label cells). Qwen3.6-27B is the strongest dense Qwen available (Apache-2.0, Apr 2026).
4. Row-pairing constraint dropped per user scope: the question is only whether the mapping transfers, so unpaired cells are acceptable.

## Design sketch (planner refines)

- **Model:** Qwen/Qwen3.6-27B only. NOTE: qwen3_5-family architecture is multimodal (image-text-to-text) — the capture rig needs a text-only port (d_model, layer count, tokenizer, vLLM support all to be verified; do NOT assume the Qwen2.5 28-layer/layer-19 conventions — sweep layers, then freeze).
- **Story generation (template-free):** seed each row with a plain-text story opening + a 2–3-scene few-shot prime fixing the target shape (one character, one in-scene question directed at them, character answers in attributed-quote dialogue), then let Qwen3.6-27B free-continue as a raw completion at temperature 1.0. The scene, its question, and the character's answer all come out of ONE untemplated pass — generation and on-policy answering merge. Judge gates: structural (exactly one question→answer exchange, parseable answer span) + basic coherence; NO congruence gate needed (congruence by construction), but score congruence anyway as a manipulation check.
- **Boundary form:** standardize on attributed quote (`{Name} replied: "…"`); the #2054 bare-label trailing-space cell is a known tokenization trap (23–44% digit-start onsets) — do not use bare-label for the headline.
- **Chat comparator:** fresh Qwen3.6-27B chat-template cells on the existing real-conversation pool (LMSYS-derived draw; reuse the #2054 sampling manifest where fit-for-purpose per artifact-reuse checklist).
- **Well-posedness:** n_train per fold must exceed d_model in the ambient basis (#2054 lesson; 6,000+ kept rows per cell at d=3,584 — resize for the 27B d_model), with the train-fold reduced-basis companion as the regime check.
- **Target framings (transfer destinations; chat is the ONLY training cell):** plain-text `User:/Assistant:`; story with an assistant-like character; story with a small panel of distinct characters varying judge-scored AI-likeness (the #1345 char-ladder shape: chat-trained map → each character cell, recovery fraction vs AI-likeness). Target-cell own maps are fit ONLY as ceiling denominators.
- **Reads:** chat→target transfer R² per target cell, reparameterization ladder vs matched-capacity nulls (one direction), recovery fraction vs character AI-likeness, identity+bias baseline, kNN retrieval (chance stated). Report per-cell answer-quality diagnostics (opener-collapse rate, digit-start rate, language mix, judge coherence) alongside the map reads so map values are interpretable against generation quality.
- **Disclosures:** fully-independent query pools ⇒ any transfer gap mixes framing with topic shift — state in every transfer read; instruct-authored fiction remains (authorship objection weakened by raw completion, not removed — the weights are still assistant-tuned).

## Provenance

Originating prompt (verbatim, user chat 2026-08-18): "can we rerun with: - a stronger model (qwen3.6 -- what are the sizes) - queries that make sense in the story - generate story without chat template -> what's the best way to do this - the queries don't have to be matched. We are just checking if the mapping transfers. Ask clarifying questions"

Clarifier answers (user, same session): model = Qwen3.6-27B only ("we don't need base model"); story source = scene-first with the query invented to fit the scene; template-free generation = instruct raw completion; query pools = fully independent, confound accepted and disclosed; transfer scope (verbatim): "The only transfer I want to check is: - trained in chat template -> all other framings including different characters".
