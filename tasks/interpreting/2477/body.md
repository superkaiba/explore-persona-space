---
title: Verify base-model completions under the chat template are coherent (paper base-row
  format check)
kind: analysis
tags: []
created_at: '2026-08-22T20:08:58Z'
has_clean_result: false
parent_id: 825
origin_prompt: 'Paper outline 2026-08-22: ''TO VERIFY: BASE MODEL COMPLETIONS IN CHAT
  TEMPLATE ARE COHERENT — ELSE USE BARE TEXT FORMAT'''
workflow: v1
---
## Goal

Verify that Qwen2.5-7B BASE-model completions generated under the chat template are coherent — the validity premise behind the paper's Results II base-model rows (#825: base map holds ~87% of instruct strength). Judge-score a sample of the banked base-arm completions (coherence/fluency 0-100, project judge, N≥5 draws per item on a few hundred items) and compare against instruct-arm completions. Decision output for the paper: if base-under-chat-template completions are NOT coherent, the paper's base rows must be re-stated on the bare-text format (persona-vectors-in-pretraining convention) — say which format each existing base-row artifact actually used.

## Provenance

Paper outline (Thomas, 2026-08-22), Results II: "TO VERIFY: BASE MODEL COMPLETIONS IN CHAT TEMPLATE ARE COHERENT — ELSE USE BARE TEXT FORMAT (following previous work on persona vectors in pretraining)". Related: #1336 has an in-flight base-coherence validity round for the Llama Tülu ladder; this task covers the Qwen #825/#2061 rows.

## Design notes

- ANALYSIS-FIRST: inventory which format (#825, #2061, #1902, #1336) each base arm actually generated under (chat template vs plain "User:/Assistant:" vs bare text) from their methodology sections + banked raw completions on the HF data repo, BEFORE generating anything new.
- Only if no base-under-chat-template completions are banked: generate a small sample (≤1 GPU-h, vLLM).
- Judge = claude-sonnet-4-5-20250929 per project rule.
