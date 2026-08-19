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
  in Qwen3.6-27B (instruct-only): plain-text, an assistant-like story character, a
  panel of distinct story characters spanning an AI-likeness gradient, and a dialogue-reply
  arm where the addressed utterance is ordinary dialogue rather than a question (does
  transfer track question-answering specifically or interlocutor-response generally);
  one-directional transfer only, stories generated fully template-free (raw completion),
  scene-native invented queries, fully independent query pools (topic-shift confound
  disclosed).'
---
# Chat-trained context→answer map transfer to story framings and characters in Qwen3.6-27B with scene-native queries and fully template-free story generation

## Goal

Test whether the context→answer linear map trained in the chat-template framing transfers — directly or up to a linear reparameterization — to every other framing in Qwen3.6-27B (instruct-only): plain-text, an assistant-like story character, a panel of distinct story characters spanning an AI-likeness gradient, and a dialogue-reply arm where the addressed utterance is ordinary dialogue rather than a question (does transfer track question-answering specifically or interlocutor-response generally); one-directional transfer only, stories generated fully template-free (raw completion), scene-native invented queries, fully independent query pools (topic-shift confound disclosed).

## Why (what this fixes over #1345/#2054)

The #1345/#2054 story cells carry four objections this rerun removes or upgrades:
1. Scene–query incongruence: #2054's scaffold rig sampled (setting, situation, register) independently of the verbatim query and gated admission only on structural extractability — Java questions in medieval monasteries were admitted by design; the resulting deflection/non-answer continuations contaminate on-policy answer targets. Here the query is invented WITH the scene, so congruence holds by construction.
2. Chat template in the generation path: prior scaffolds were written by the instruct model under a templated instruction (tier-2 instruct-and-strip). Here generation is raw completion — no template tokens ever exist, even upstream.
3. Model strength: Qwen2.5-7B continuations showed mode collapse ("Being" openers 60% in one base cell; digit slot-filling 44%/19% base/instruct in bare-label cells). Qwen3.6-27B is the strongest dense Qwen available (Apache-2.0, Apr 2026).
4. Row-pairing constraint dropped per user scope: the question is only whether the mapping transfers, so unpaired cells are acceptable.

## Design sketch (planner refines)

- **Model:** Qwen/Qwen3.6-27B only. NOTE: qwen3_5-family architecture is multimodal (image-text-to-text) — the capture rig needs a text-only port (d_model, layer count, tokenizer, vLLM support all to be verified; do NOT assume the Qwen2.5 28-layer/layer-19 conventions — sweep layers, then freeze).
- **Story generation (template-free), three-segment recipe — structure is guaranteed by prefill at the boundaries, never by instruction:**
  - *Segment A (scene + question, shaped then mined):* prompt = 2–3-example few-shot prime in the exact target shape (prose scene → one quoted question to the character → `{Name} replied: "…"` answer → close, `***` separators) + a fresh opening seed (setting/situation/register sampled; final seed sentence from a varied question-imminent bank). Free-continue at temperature 1.0; keep the first quoted `?`-utterance directed at the character; drop rows with no question within ~250 tokens. Precedent: #1310 base script scenes (3-shot prime, 0.90 attribution-precision gate PASS on a far weaker model).
  - *Segment B (answer, forced):* truncate right after the question, append the attribution opener `\n\n{Name} replied: "` (phrasing varied over a small bank within the attributed-quote family — a frozen string is the #1345 hardcoded-template trap, R² 0.019), free-continue to the closing quote. Answer existence + span are by construction; the opener IS the measured boundary form.
  - *Judge gate (filter, not force):* exactly one question, question directed at the character, basic coherence; congruence scored only as a manipulation check (near-ceiling expected by construction). Oversample vs a pre-registered yield floor (#1345 on-policy kept 59% of attempts with a templated instruction — the prime+prefill band should match or beat it); shortfall reported, never backfilled.
  - *Disclosures:* the few-shot prime is stripped before capture (activations read over the standalone story text, so measured context ≠ generation context — the untemplated analogue of instruct-and-strip); the forced attribution opener means the answer onset is partially experimenter-written, coinciding with the standardized boundary form.
- **Boundary form:** standardize on attributed quote (`{Name} replied: "…"`); the #2054 bare-label trailing-space cell is a known tokenization trap (23–44% digit-start onsets) — do not use bare-label for the headline.
- **Chat comparator:** fresh Qwen3.6-27B chat-template cells on the existing real-conversation pool (LMSYS-derived draw; reuse the #2054 sampling manifest where fit-for-purpose per artifact-reuse checklist).
- **Well-posedness:** n_train per fold must exceed d_model in the ambient basis (#2054 lesson; 6,000+ kept rows per cell at d=3,584 — resize for the 27B d_model), with the train-fold reduced-basis companion as the regime check.
- **Target framings (transfer destinations; chat is the ONLY training cell):** plain-text `User:/Assistant:`; story with an assistant-like character; story with a small panel of distinct characters varying judge-scored AI-likeness (the #1345 char-ladder shape: chat-trained map → each character cell, recovery fraction vs AI-likeness). Target-cell own maps are fit ONLY as ceiling denominators.
- **Dialogue-reply arm (user-requested):** a parallel story cell family where the utterance directed at the character is ORDINARY DIALOGUE, not a question — a statement, remark, or confidence the character then responds to. Same three-segment recipe: Segment A mines the first quoted non-`?` utterance directed at the character (judge additionally rejects questions-in-disguise: indirect/rhetorical interrogatives), Segment B forces the reply with the same varied attribution opener; context slot = end of the addressed utterance (the structural analogue of the chat query-end slot). Discriminating read: if the chat-trained map transfers to statement-replies about as well as to question-answers, the map is a general respond-to-the-interlocutor object; if it collapses specifically on non-questions, it is a question→answer content map. Report the question vs non-question transfer gap per character alongside the AI-likeness gradient.
- **Reads:** chat→target transfer R² per target cell, reparameterization ladder vs matched-capacity nulls (one direction), recovery fraction vs character AI-likeness, identity+bias baseline, kNN retrieval (chance stated). Report per-cell answer-quality diagnostics (opener-collapse rate, digit-start rate, language mix, judge coherence) alongside the map reads so map values are interpretable against generation quality.
- **Disclosures:** fully-independent query pools ⇒ any transfer gap mixes framing with topic shift — state in every transfer read; instruct-authored fiction remains (authorship objection weakened by raw completion, not removed — the weights are still assistant-tuned).

## Provenance

Originating prompt (verbatim, user chat 2026-08-18): "can we rerun with: - a stronger model (qwen3.6 -- what are the sizes) - queries that make sense in the story - generate story without chat template -> what's the best way to do this - the queries don't have to be matched. We are just checking if the mapping transfers. Ask clarifying questions"

Clarifier answers (user, same session): model = Qwen3.6-27B only ("we don't need base model"); story source = scene-first with the query invented to fit the scene; template-free generation = instruct raw completion; query pools = fully independent, confound accepted and disclosed; transfer scope (verbatim): "The only transfer I want to check is: - trained in chat template -> all other framings including different characters".
