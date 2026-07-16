---
title: 'Which property scopes the shared assistant context→answer map: helpful register,
  or speaking to a user?'
kind: experiment
tags: []
created_at: '2026-07-16T08:28:52Z'
has_clean_result: false
parent_id: 825
origin_prompt: 'Help me to test these hypotheses: [the #825 chat-vs-no-template-vs-story
  writeup + Next Steps hypotheses — ''this mapping is only for when the assistant
  is being helpful'' vs ''this mapping is only for when the assistant is speaking
  to a user''; verbatim full prompt in ## Provenance]'
workflow: v1
goal: 'On Qwen-2.5-7B base + instruct, determine which property of the assistant-chat
  setting the shared context→answer map is tied to — the assistant''s helpful register
  (H1) or the assistant speaking to a user (H2) — by fitting the frozen #825 ridge
  recipe (held-out conv-grouped K-fold; prefix-based AND context-based arms) on on-policy
  answers to the same 4724-conv LMSYS pool under framing cells that decouple the two
  properties (rude-but-informative dialogue, evasive dialogue, helpful-instruction
  control, addressee-free exposition, non-user-addressee dialogue), reading map identity
  against the chat-template reference via the #825 similarity battery (bidirectional
  transfer R², linear reparameterization recovery, raw + rotation-aligned cosine).'
---
# Which property scopes the shared assistant context→answer map: helpful register, or speaking to a user?

## Goal

On Qwen-2.5-7B base + instruct, determine which property of the assistant-chat setting the shared context→answer map is tied to — the assistant's helpful register (H1) or the assistant speaking to a user (H2) — by fitting the frozen #825 ridge recipe (held-out conv-grouped K-fold; prefix-based AND context-based arms) on on-policy answers to the same 4724-conv LMSYS pool under framing cells that decouple the two properties (rude-but-informative dialogue, evasive dialogue, helpful-instruction control, addressee-free exposition, non-user-addressee dialogue), reading map identity against the chat-template reference via the #825 similarity battery (bidirectional transfer R², linear reparameterization recovery, raw + rotation-aligned cosine).

## Overview / Motivation

The #825 line established:

1. The context→answer map is the SAME with and without the chat template, up to a linear change of coordinates (reparameterization recovers it; rotation-aligned cosine ~0.73–0.85; instruct raw cosine 0.65 vs 0.29 base — instruct-tuning rotates the two framings' maps together). (#825 Result 3 + the reparameterization round.)
2. The map is NOT the same when the assistant is a character in a story: transfer fails both directions and reparameterization does not recover it. (#825 Result 4; refined by Result 4.5 / #1310 — per-character story maps EXIST once properly powered, but they are *different, character-specific* maps.)

So the shared map is scoped to something about the assistant-answering-a-user setting. Chat/no-template and story framings confound at least two candidate properties, which this task decouples:

- **H1 (helpfulness):** the map is tied to the assistant *being helpful* (the helpful-answer register/function).
- **H2 (user-directedness):** the map is tied to the assistant *speaking to a user* (direct second-person dialogue with an interlocutor), regardless of register.

Secondary outcomes are also informative: BOTH properties required (conjunction), or NEITHER (the map is generic direct-answer/QA structure — then the story divergence is about the narrative frame itself, connecting to #1310's character-specific maps).

## Design

Single manipulated variable family: the FRAMING of the same query pool. Same 4724-conversation single-turn LMSYS pool as #825 (reuse); all answer text generated ON-POLICY per cell per model (base + instruct); no training anywhere — generation + activation capture + regression fits only (#825 recipe).

| Cell | Helpful register | User-directed dialogue | Purpose |
|---|---|---|---|
| C0 chat-template assistant (REUSE #825) | yes | yes | reference map |
| C0′ no-template "User:/Assistant:" (REUSE #825) | yes | yes | reference replication |
| C1 helpful-instruction system prompt | yes | yes | instruction-PRESENCE control (matches C2/C3's added system prompt so the manipulated variable is register content, not instruction presence) |
| C2 rude-but-informative assistant (system-prompted: answers correctly but hostile/begrudging) | **no** | yes | load-bearing H1 cell — breaks register, HOLDS answer content |
| C3 evasive/unhelpful assistant (deflects, refuses to engage) | **no** | yes | strong helpfulness break; carries the content-collapse caveat (see Measurement) |
| C4 addressee-free exposition (query rendered as a topic/heading; model continues an encyclopedic passage; no dialogue tokens, no second person) | yes | **no** | load-bearing H2 cell |
| C5 non-user addressee dialogue (another AI/colleague asks the same question; direct dialogue held) | yes | partial | refines H2: "a user" vs "any interlocutor" |

**Prediction table:**

| Outcome | C2 (rude) vs ref | C4 (exposition) vs ref |
|---|---|---|
| H1 (helpful-only) | different | same (up to linear reparam) |
| H2 (user-directed-only) | same | different |
| Conjunction | different | different |
| Neither (generic QA) | same | same |

## Measurement

- **Map existence per cell:** held-out conv-grouped CV R² vs the shuffle-answer permutation null (frozen #825 recipe: per-cell lambda-grid dual ridge, layer 19 headline + frozen layer set {14,18,19} + sweep subsample).
- **Map identity vs reference (the #825 battery, per cell × model):** (a) bidirectional transfer R²; (b) linear reparameterization recovery — learn A_ctx (paired same-query contexts) and A_ans (paired same-query answers), test M_ref ≈ A_ans · M_cell · A_ctx⁻¹; (c) flattened-map raw cosine + best-rotation-aligned cosine (`issue825_map_alignment.py`).
- **Both mapping arms, per the standing rule:** prefix-based (prefix = everything before the user query: system prompt / framing preamble) AND context-based (prefix + query). Per-cell span definitions (what counts as "query" in C4's topic rendering) fixed at plan time.
- **Register-compliance judge filter:** every generated answer judged for cell compliance (graded 0–100, `claude-sonnet-4-5-20250929`, drop-never-coerce, per `.claude/rules/llm-judging.md`); keep compliant rows, report per-cell yield. A below-floor cell is reported, never silently backfilled.
- **Content-collapse diagnostics (the key confound):** an evasive cell can "break the map" trivially because answers stop depending on the query (nothing left to predict). Report per-cell answer-target variance and the shuffle-null band width; the load-bearing H1 read comes from C2 (content held), with C3 as the gradient point. A C2 map failure with matched target variance is the clean H1 signal.

## Reuse (fitness-check at plan time per `.claude/rules/artifact-reuse.md`)

- 4724-conv LMSYS pool + chat-template / no-template captures + fitted reference maps: #825 rounds (HF data repo `issue825_*`).
- Code: `issue825_render_formats.py` (framing renderer — extend with C1–C5 renderings), `issue825_gen_conversations.py` / on-policy gen scripts, `issue825_fit_cells.py` (grouped-CV dual ridge), `issue825_map_alignment.py` (reparameterization + rotation alignment), `issue825_crossmodel_map_transfer.py` (transfer).
- External comparison: #1310's per-character story maps (base Wren 0.137 / HELIOS 0.148 / …) once its instruct control lands.

## Estimated cost

~5 new cells × 2 models × (on-policy generation over 4724 queries + activation capture) + fits (batched dual-space, CPU/GPU-cheap). By #825 round analogy ≈ 2–4 GPU-h per cell-model on A100 ⇒ **~25–35 GPU-h** (planner refines; wide GCP provisioning preferred, cells shard cleanly).

## Provenance

- Filed from user chat 2026-07-16. Routed as a CHILD of #825 (question_relation: substantially-different — scope conditions of the map would change the parent Goal; precedent: #1310 filed the story-character framing question as a child). NOT auto-spawned (filing ≠ spawning).
- Verbatim originating prompt:

> Help me to test these hypotheses:
> Result - The assistant map is the same with and without the chat template (up to a linear change of coordinates) - it is not the same as when the assistant is a story character
> Motivation
> * We've found a mapping from assistant to assistant answer with the chat template and without the chat template
> * We've also found that there is no mapping from story character to story character dialogue
> * Here I wanted to test if the mapping from assistant to assistant answer is the same: with the chat template / without the chat template / in story format
> Methodology
> * 4724 conversations from LMSYS (same as before)
> * 2108 answer-turns from 420 on-policy-generate stories for the story framing
> * Framed: with chat template (standard Qwen chat template) / without chat template (same conversations rendered as "User: + Assistant:") / in story format (an AI character answering a human's questions in narrative prose)
> * [Dashboard with examples (replace ARIA with assistant)]
> Results
> Result 1: The assistant map is the same with and without the chat template (up to a linear change in coordinates)
> * I first wanted to directly test if the assistant map is the same with and without the chat template (up to a linear change in coordinates)
> Methodology
> * Fit the ridge regression map (context → answer) in each framing
> * Run transfer from chat template → no template
> * Reparameterize one map into the other's coordinates and check if recovers mapping: Learn A_ctx → x_chat = A_ctx x_notemplate, and A_ans → y_ans = A_ans y_notemplate; Reparameterize M_chat = A_ans M_notemplate A_ctx^-1; similarly for M_notemplate
> Takeaways
> * After reparameterization, the chat and no template maps are almost identical (in both base and instruct models)
> I then wanted to see how similar these 2 maps were (without reparameterization)
> Methodology: Fit the base/instruct context → answer maps with and without chat template; for each of base, instruct: flatten the 2 maps, take cosine similarity → raw cosine similarity; do best rotation to align matrices → rotation-aligned cosine similarity; look at how similar they are before and after rotation-alignment
> Takeaways
> * The 2 maps are quite similar when rotation aligned (cosine similarity ~0.73-0.85)
> * Instruct finetuning rotates the 2 maps to be closer together (instruct raw cosine similarity of 0.65 vs 0.29 in base model)
> Result 2: The assistant map is not the same when the assistant is a character in a story
> * I then wanted to test if this was the same mapping as when the assistant was a character in a story
> Methodology
> * Fit the ridge regression map (context → answer) in chat template vs in story framing
> * Run transfer from chat template → story and story → chat template
> * Reparameterize one map into the other's coordinates and check if recovers mapping
> Takeaways
> * The mapping is completely different when the assistant is a story character
> Conclusion/Next Steps
> * This assistant context → answer mapping is restricted to the assistant speaking to a user? and does not transfer to story framing — slight hit to PSM which might predict these mappings would be the same
> * Hypotheses: this mapping is only for when the assistant is being helpful; this mapping is only for when the assistant is speaking to a user; testing these
