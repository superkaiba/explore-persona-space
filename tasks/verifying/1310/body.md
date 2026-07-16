---
title: Focused per-character context->dialogue map on on-policy stories, base vs instruct
  (4-persona panel)
kind: experiment
tags: []
created_at: '2026-07-14T22:56:43Z'
has_clean_result: false
parent_id: 931
origin_prompt: i just want a mapping focused on a specific character in a fiction
  story (NOT the assistant) - trained on on policy generated stories, in base model
  and in instruct model. go with the 4-persona panel. Make sure each persona always
  uses the same label
workflow: v1
goal: Test whether a context->dialogue linear map focused on a SINGLE fixed fiction
  character (persona, NOT the assistant), trained on on-policy generated stories that
  always refer to the character by the same fixed label, exists in Qwen2.5-7B base
  and instruct — the fiction-character analog of the assistant-persona context->answer
  map — across a 4-persona panel, each persona its own map.
---
## Goal

Test whether a context->dialogue linear map focused on a SINGLE fixed fiction character (persona, NOT the assistant), trained on on-policy generated stories that always refer to the character by the same fixed label, exists in Qwen2.5-7B base and instruct — the fiction-character analog of the assistant-persona context->answer map — across a 4-persona panel, each persona its own map.

## Motivation

The fiction arm of #931 pooled ACROSS characters (each character collapsed to one aggregated point, one map over all of them) and found no transfer of the chat map to fiction — but that design never tested a map FOCUSED on one character with many data points. This tests the natural per-character analog: fix a character (a persona with a stable name), generate many on-policy stories where it speaks, and fit context→that-character's-dialogue across those stories — many points, one character — the direct fiction mirror of "one assistant persona → many answers." Run in base AND instruct to locate where (if anywhere) a character-persona map lives.

## Design

- **Models:** Qwen2.5-7B (base) + Qwen2.5-7B-Instruct, compared on identical generated text where possible (each model generates its own on-policy stories; the map is fit per model).
- **4-persona panel, each with a FIXED LABEL used in every story + in attribution** (stable-label requirement — the character is always referred to by exactly this name so dialogue turns are reliably located):
  1. `Marlowe` — a hardboiled 1940s private detective
  2. `Pip` — a cheerful, curious young child
  3. `Bexley` — a formal, deferential butler
  4. `HELIOS` — a calm sci-fi starship AI
- **On-policy story generation** (sampled decoding T=1.0, top-p 0.95 — NOT greedy; base greedy degenerates into loops on raw prose, #825 r7/8): per persona, generate N stories (target ~250/persona/model) where the character named by the fixed label speaks multiple times. Base: raw-prose completion prompt naming the character; instruct: chat-template request to write the story. The character's name is pinned to the fixed label in the prompt so every generation uses it.
- **Extraction:** attribute the persona's dialogue turns by the fixed label (dialogue-tag extractor keyed on `<LABEL>`, judge-audited precision floor 0.90 as in #931). Per story → one (X, Y) point for that persona: X = mean activation over the story context before the persona's dialogue (setup + the character's turn cue), Y = mean activation over the persona's attributed dialogue. Teacher-forced 28-layer capture, bf16.
- **Fit (per persona, the focused map):** per-layer GCV Gram ridge, K=5 STORY-grouped folds (seed 0), frozen layers {14,18,19,26} headline 19, 20 shuffle nulls + 1000-draw bootstrap — fit and held-out-tested WITHIN one persona's stories (many points, one character). One map per persona per model.
- **Reads:** (a) does the focused per-persona map exist (held-out R² above the shuffle null) in base and instruct; (b) base-vs-instruct strength per persona; (c) character-specificity: predict persona A's dialogue from persona B's context (swap) vs correct; (d) reference: the same model's assistant map (#825 committed S1/S2) as a ceiling.

## Reuse
Reuse #931's story-generation + dialogue-attribution + extraction machinery and the issue825_fit_cells / issue931 fit core; the only genuinely new pieces are the fixed-label persona prompts and the per-persona (within-character) fit grouping.

## Success / kill
Success: at least one persona shows a per-character map with held-out R² clearing its shuffle null in at least one model, with base-vs-instruct reported. Kill: all per-persona maps at/below the shuffle null in both models (a focused character map does not exist even with many same-character points) — itself a clean negative that sharpens #931.

## Compute
On-policy generation (2 models × ~1000 stories, sampled) + teacher-forced extraction + closed-form fits. Est ~2–3 GPU-h. Base story generation sampled (not greedy). Provenance: on-policy, both models; per-arm stated.
