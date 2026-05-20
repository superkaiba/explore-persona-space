---
title: 'Aim 4.2: Check if FineWeb contains AI chat data'
kind: infra
tags: []
created_at: '2026-04-16T19:30:25.000Z'
has_clean_result: false
sagan_id: 95035779-d06f-4188-8bd6-12c314590b85
sagan_number: 22
priority: normal
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

Search FineWeb-Edu (and/or raw FineWeb) for documents that look like AI assistant chat data (ChatGPT/Claude/Bard transcripts, "As an AI language model" patterns, chat-format Q&A).

Quantify prevalence: what fraction of high-axis-projection docs are actually AI chat transcripts?

**Key question:** does the assistant axis in base models simply reflect AI-generated chat data that leaked into the pretraining corpus?

- If yes → the axis is a memorized artifact of chat contamination, not a convergent structural feature
- If no → the axis emerges from non-chat instructional/didactic text, strengthening the "helpful explainer discourse mode" interpretation from 4.2

Method: keyword search + Claude classifier on top/bottom 500 axis-projection docs from existing 200K FineWeb-Edu projections.

Compute: ~1h analysis, minimal GPU (reuse existing projections). Pod: any or local.
