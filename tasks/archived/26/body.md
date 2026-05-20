---
title: 'Aim 4.5: Random direction control for category rankings'
kind: infra
tags: []
created_at: '2026-04-16T19:30:29.000Z'
has_clean_result: false
sagan_id: 9d9d8a85-1bf9-4dea-8f2d-f613953e61dc
sagan_number: 26
priority: normal
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

Project 200K FineWeb-Edu + 200K LMSYS onto 10 random unit vectors in Qwen3-32B layer 32 (5120-D).

For each direction: take top/bottom 200 docs, classify with same taxonomy (genre, author stance).

Also project 3,514 category docs (from category projection experiment) onto same 10 directions, compare median rankings.

**Key question:** do the 4 BH-surviving findings (Genre + Author Stance × 2 corpora) appear on random directions too?

- If yes → axis is not special, category structure is generic high-D geometry
- If no → axis captures genuinely specific discourse mode structure

Compute: ~90 min model inference (speculators+vLLM on H200) + ~30-60 min Claude classification (~4K docs) + analysis.
Pod: thomas-rebuttals (4x H200). Estimated cost: ~$5-15 Claude API.
