---
title: Causal patching of context vectors across framings (chat-to-plain, assistant-to-story)
kind: experiment
tags: []
created_at: '2026-08-19T03:57:22Z'
has_clean_result: false
parent_id: 2094
origin_prompt: ok file these but don't run them right away (paper C4 causal upgrade;
  claims.md Gap 1)
workflow: v1
goal: 'Test whether the framing-transfer results (#2054 chat-plain, #1345/#1639 assistant-story)
  are causal: patch the context vector v_C across framings and measure persona/behavior
  transport into the target framing''s generated answer vs matched within-framing
  patches and null controls.'
---
# Causal patching of context vectors across framings (chat→no-template, assistant→story)

## Goal

Test whether the framing-transfer results (#2054 chat-plain, #1345/#1639 assistant-story) are causal: patch the context vector v_C across framings and measure persona/behavior transport into the target framing's generated answer vs matched within-framing patches and null controls.

## Context

- Paper need: C4 (PSM section) of the context→answer mapping paper (`~/overleaf-6a59c927/plan.md`; `claims.md` Gap 1). C4 is currently correlational-only; these two arms are what upgrade it to causal.
- Method inheritance: #2094's null-separated patching grid (context-end patches, fraction-of-full-swap DV, bootstrap screen + independent temperature-1.0 confirmation). #2333's prefill-vs-patch decomposition (opening-token share of the patch effect) qualifies interpretation and should be carried as a control.
- Two arms: (a) chat-template ↔ no-template ("User:/Assistant:" plain form) patches; (b) assistant ↔ story-character patches on the constructed rigs from #1345/#1639.
- Deadline context: ICLR 2027 abstracts 2026-09-18.

## Scheduling

NOT SCHEDULED — filed per user directive 2026-08-19 ("file these but don't run them right away"). Do not auto-run; user/PM dispatches.
