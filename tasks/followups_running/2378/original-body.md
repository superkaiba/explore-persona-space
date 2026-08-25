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
backend: runpod
goal: 'Test whether the context→answer linear map trained in the chat-template framing
  (assistant turn) transfers — directly or up to a linear reparameterization — to
  every other framing in Qwen3.6-27B (instruct-only): plain-text, an assistant-like
  story character, a panel of distinct story characters, a dialogue-reply arm where
  the addressed utterance is ordinary dialogue rather than a question, and the USER
  character inside the chat template (two provenance arms: real human user turns teacher-forced,
  and simulated user turns the model writes on-policy; context slot ends at the previous
  assistant turn so the target is never self-predicted); one-directional transfer
  only, stories generated fully template-free (raw completion), scene-native invented
  queries, fully independent query pools (topic-shift confound disclosed). Secondary
  arm: one general map fit pooled on ALL cells compared per cell to the specialized
  own maps (pooled-tier ladder). Every arm reports BOTH held-out R² and rank-1 retrieval
  under the #2202 conventions (whitened cosine, CSLS, convention-matched fresh-draw
  reference). No AI-likeness judge axis (dropped per user directive).'
---
# Chat-trained context→answer map transfer to story framings and characters in Qwen3.6-27B with scene-native queries and fully template-free story generation

## Goal

Test whether the context→answer linear map trained in the chat-template framing (assistant turn) transfers — directly or up to a linear reparameterization — to every other framing in Qwen3.6-27B (instruct-only): plain-text, an assistant-like story character, a panel of distinct story characters, a dialogue-reply arm where the addressed utterance is ordinary dialogue rather than a question, and the USER character inside the chat template (two provenance arms: real human user turns teacher-forced, and simulated user turns the model writes on-policy; context slot ends at the previous assistant turn so the target is never self-predicted); one-directional transfer only, stories generated fully template-free (raw completion), scene-native invented queries, fully independent query pools (topic-shift confound disclosed). Secondary arm: one general map fit pooled on ALL cells compared per cell to the specialized own maps (pooled-tier ladder). Every arm reports BOTH held-out R² and rank-1 retrieval under the #2202 conventions (whitened cosine, CSLS, convention-matched fresh-draw reference). No AI-likeness judge axis (dropped per user directive).

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
- **Target framings (transfer destinations; chat is the ONLY training cell):** plain-text `User:/Assistant:`; story with an assistant-like character; story with a small panel of distinct characters (the #1345 char-ladder shape: chat-trained map → each character cell, per-character recovery fraction; NO AI-likeness judge axis — dropped per user directive 2026-08-19). Target-cell own maps are fit ONLY as ceiling denominators.
- **User-character-in-chat-template framing (user-requested), two provenance arms:** the USER role inside the standard chat template as a transfer destination — (a) REAL user turns: teacher-forced capture over the actual human user turn from the conversation pool; (b) SIMULATED user turns: the measured model writes the user's turn on-policy (prefill the template through the end of the previous assistant turn + the user-turn header; the model continues as the user). SLOT DEFINITION IS LOAD-BEARING (#2054 excluded the user identity precisely because its rig's context slot ended at the query, i.e. AT the user turn — self-prediction): here the context slot ends at the end of the PREVIOUS ASSISTANT TURN (+ user-turn header), target = user-turn mean activation, so the target text is never inside the context. Requires multi-turn conversations (≥1 completed assistant turn before the measured user turn). Recipe precedent: #825's user-turn cells. #825's CORRECTED result (selector-audit settle battery, folded into its body 2026-08-19): the earlier user-turn linear null was an unguarded-GCV estimator artifact at n_tr < d — under the guarded selector (dof cap 0.9, the estimator this plan inherits) the 7B user cells are weakly linearly predictable (ridge R² ≈ +0.19…+0.25 against a ~0.78 resample ceiling, ~2.5× below the assistant). This arm tests whether the weak 7B read strengthens, holds, or regresses on a 4× model. Read: chat-assistant-trained map → user cells, both provenance arms, same dual R²+retrieval reporting; the real-vs-simulated gap is itself a read (does the map track the model's simulation of the user or the human distribution).
- **Dialogue-reply arm (user-requested):** a parallel story cell family where the utterance directed at the character is ORDINARY DIALOGUE, not a question — a statement, remark, or confidence the character then responds to. Same three-segment recipe: Segment A mines the first quoted non-`?` utterance directed at the character (judge additionally rejects questions-in-disguise: indirect/rhetorical interrogatives), Segment B forces the reply with the same varied attribution opener; context slot = end of the addressed utterance (the structural analogue of the chat query-end slot). Discriminating read: if the chat-trained map transfers to statement-replies about as well as to question-answers, the map is a general respond-to-the-interlocutor object; if it collapses specifically on non-questions, it is a question→answer content map. Report the question vs non-question transfer gap per character.
- **Pooled-vs-specialized arm (user-requested):** alongside the chat-trained transfer, fit ONE general map on EVERYTHING pooled (all framings, characters, dialogue-reply, and chat rows together) and compare it to each cell's specialized own map — the #2054 pooled-tier ladder shape (pooled as-is → + per-cell bias → + low-rank per-cell residual), reporting per-cell recovery fraction of the specialized ceiling at each tier. This answers generalization-vs-specialization (the pooled map SEES target-framing rows), complementary to the transfer arm (the chat map never does); both arms run on the same cells.
- **Reads — R² AND retrieval, both metrics on every arm (user directive; retrieval conventions from #2202):** per target cell and per transfer/pooled rung, report BOTH held-out R² and rank-1 retrieval accuracy against the held-out answer pool. Retrieval follows the #2202 conventions: whitened-cosine similarity (the convention-matched metric — raw cosine understates), CSLS hub-penalized rescoring (K=10) alongside plain nearest-neighbor, a convention-matched fresh-draw reference as the retrieval ceiling, and (where multi-draw answers exist) the multi-draw-averaged target read (#2202: the map predicts the noise-averaged answer; 5-draw averaging lifted rank-1 0.815→0.909 raw). The two metrics are reported side by side because they dissociate in both directions (#722, #779, #2202: R²-poor maps can retrieve well and vice versa). Plus: reparameterization ladder vs matched-capacity nulls (one direction), per-character recovery fractions, identity+bias baseline, chance rates stated. Report per-cell answer-quality diagnostics (opener-collapse rate, digit-start rate, language mix, judge coherence) alongside the map reads so map values are interpretable against generation quality.
- **Disclosures:** fully-independent query pools ⇒ any transfer gap mixes framing with topic shift — state in every transfer read; instruct-authored fiction remains (authorship objection weakened by raw completion, not removed — the weights are still assistant-tuned).

## Provenance

Originating prompt (verbatim, user chat 2026-08-18): "can we rerun with: - a stronger model (qwen3.6 -- what are the sizes) - queries that make sense in the story - generate story without chat template -> what's the best way to do this - the queries don't have to be matched. We are just checking if the mapping transfers. Ask clarifying questions"

Clarifier answers (user, same session): model = Qwen3.6-27B only ("we don't need base model"); story source = scene-first with the query invented to fit the scene; template-free generation = instruct raw completion; query pools = fully independent, confound accepted and disclosed; transfer scope (verbatim): "The only transfer I want to check is: - trained in chat template -> all other framings including different characters".
