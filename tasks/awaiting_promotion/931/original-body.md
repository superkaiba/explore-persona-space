---
title: Story-character generalization of the context→answer map (no chat template)
  + separator-token specificity control
kind: experiment
tags: []
created_at: '2026-07-03T12:10:25Z'
has_clean_result: false
parent_id: 825
origin_prompt: 'We found a linear mapping from context to answer

  We want to see if this is a general character -> behavior mapping

  So we want to look at random **stories** inserted into the instruct model with no
  chat template

  or tell the assistant to write a story with characters talking


  and see if:

  - there is a similar linear mapping

  - how similar it is to the one we found from context to answer with the chat template


  also to test if this is a special mapping we also want to potentially test other
  random separator/puncutation tokens that have nothing to do with personas and see
  if they predict the mean activation afterwards until the next separator (probably
  punctuation)'
workflow: v1
goal: 'Test whether the linear context→answer-profile map (#779/#825 held-out ridge
  recipe, c_x → v(x)) generalizes to a character→behavior map in STORIES on Qwen2.5-7B-Instruct
  — (a) real stories fed with NO chat template, predicting the mean activation of
  a character''s dialogue span from that character''s context vector, and (b) assistant-written
  multi-character stories under the chat template — quantifying similarity to the
  chat-template context→answer map via cross-regime transfer R², matched-layer map/weight
  similarity, and layer-profile rank correlation; specificity control: the same machinery
  anchored on random persona-irrelevant separator/punctuation tokens (separator activation
  → mean activation of the following span until the next separator) to test whether
  span-mean predictability is character/persona-specific or a generic delimiter-span
  property.'
relates_to:
- identity-cb-duality
- identity-persona-vs-behavior
---
# Story-character generalization of the context→answer map + separator-token specificity control

## Goal

Test whether the linear context→answer-profile map (#779/#825 held-out ridge recipe, c_x → v(x)) generalizes to a character→behavior map in STORIES on Qwen2.5-7B-Instruct — (a) real stories fed with NO chat template, predicting the mean activation of a character's dialogue span from that character's context vector, and (b) assistant-written multi-character stories under the chat template — quantifying similarity to the chat-template context→answer map via cross-regime transfer R², matched-layer map/weight similarity, and layer-profile rank correlation; specificity control: the same machinery anchored on random persona-irrelevant separator/punctuation tokens (separator activation → mean activation of the following span until the next separator) to test whether span-mean predictability is character/persona-specific or a generic delimiter-span property.

## Overview / Motivation

The line #779 → #825 → #810 established a strong linear mapping from a context representation to the model's answer profile in the chat regime (#779: held-out R² ≈ 0.67 at layer 19 on instruct; #825: present in pretrained Qwen2.5-7B at ~87% of instruct strength, and survives naturalistic transcript formatting; #810: the mean-over-answer summary carries the map across genres). Open question this task answers: is that map a **general character → behavior mapping**, or an artifact of the assistant/chat structure? And is it **special** to persona/character-bearing contexts at all, or a generic delimiter-span prediction property of the residual stream?

## Proposed design (sketch — planner formalizes + refines)

Model: Qwen2.5-7B-Instruct (same as parent line). No training; activation extraction + generation + ridge fits.

**Arm A — raw stories, no chat template.** Feed real, dialogue-rich stories (tier-1/2 corpus per the data-realism rule — e.g. WritingPrompts / Project Gutenberg fiction; planner picks and pins) into the instruct model as raw text (no chat template). Per character: context vector `c` from the character's introduction/description span (extraction recipe — boundary-token vs mean — formalized at planning, cf. #658/#920) → target `v` = mean activation over that character's dialogue span(s) (mean summary per #810). Fit the #779 held-out ridge per layer (K-fold, GCV λ, skill/R² vs selection-symmetric shuffle nulls per #778).

**Arm B — assistant-written stories, chat template.** Instruct the assistant to write a story with characters talking; run the same character-context → character-dialogue-span extraction and fit on the generated stories.

**Similarity read — is it the SAME map?** Quantify similarity of each story map to the chat-template context→answer map:
- cross-regime transfer R² (frozen chat map applied to story pairs and vice versa, reported against the within-regime refit ceiling);
- matched-layer weight/subspace similarity of the ridge maps (e.g. principal angles / CKA);
- layer-profile rank correlation (#825's replication-gate statistic).

**Specificity control — separator/punctuation tokens.** On text with no persona content, anchor the same machinery on random separator/punctuation tokens: activation at the separator → mean activation of the following span until the next separator (probably the next punctuation). If this generic delimiter→span map matches the character map's strength and transfers onto it, the context→answer map is generic span-prediction, not a persona/character-special mechanism.

## Hypotheses

- **H1 (generality):** a character→dialogue-profile map exists in raw stories without the chat template, at strength comparable to the chat-regime map (cf. #825's naturalistic-format arm).
- **H2 (shared map):** the story map is substantially the SAME map — high cross-regime transfer relative to the within-regime ceiling — consistent with one general character→behavior mechanism.
- **H3 (specificity):** the separator→span control map is weaker and/or does not transfer onto the character map. Competing hypothesis: generic span-mean predictability (topic persistence / local-context smoothness) explains most of the map — cf. #825's topic-persistence baseline and #922's per-position next-token activation map, which is the position-level analogue of this control.

## Constraints / notes

- **Formalization-first (standing rule):** before any code, the planner defines EXACTLY (1) the character-context vector and the character-behavior span target in stories (including the dialogue-attribution recipe — which utterances belong to which character), (2) what counts as "the same map" (the similarity measure + its null), and (3) the baseline family (topic-persistence, separator control) — plus a thorough lit review naming the closest prior formalizations (character/persona representations in fiction, simulacra/role-play readings of assistant behavior).
- **Reuse:** #779 `fit_h` ridge machinery; #825 naturalistic-format code paths + turnstore; #810 mean-summary recipe; #778 selection-symmetric nulls. Vectorize all many-cell fits (`.claude/rules/vectorize-many-cell-fits.md`).
- **Related in-flight work:** #922 (next-token activation prediction, running) is the per-position analogue of the separator control; #920 (context×answer summary-recipe sweep, running) may update the summary recipes — planner checks their landed takeaways before pinning recipes.
