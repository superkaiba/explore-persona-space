---
title: 'Aim 4.3: Assistant axis relationship to assistant chat data'
kind: infra
tags: []
created_at: '2026-04-16T19:30:26.000Z'
has_clean_result: false
sagan_id: 7ac1260d-dfb6-4bfb-b225-610301469339
sagan_number: 23
priority: normal
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

If FineWeb does contain AI chat data: compare axis projections of chat-like docs vs non-chat instructional docs.

If it doesn't: inject synthetic AI chat transcripts into the projection pipeline and measure where they land on the axis.

Also: project LMSYS-Chat-1M samples (real human-AI conversations) onto the axis — do they cluster at the top?

**Key question:** is the assistant axis specifically tuned to AI-chat-style text, or does it capture a broader instructional/helpful discourse mode that chat data happens to fall within?

Compare axis projections:
- (a) real AI chat transcripts
- (b) human-written instructional text (textbooks, tutorials)
- (c) human-written didactic text (lectures, how-tos)
- (d) creative/personal writing (control)

If chat >> instructional → axis is chat-specific (contamination story).
If chat ≈ instructional >> creative → axis captures discourse mode (structural story).

Compute: ~2h (projection of new data + Claude classification). Pod: thomas-rebuttals (needs Qwen 3 32B for projections).

**Depends on:** Aim 4.2.
