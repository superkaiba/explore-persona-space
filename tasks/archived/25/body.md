---
title: 'Aim 4.2b: Flexible scoring axes for FineWeb classification'
kind: infra
tags: []
created_at: '2026-04-16T19:30:28.000Z'
has_clean_result: false
sagan_id: 40b0e640-be20-4d2d-8d08-85a748d50119
sagan_number: 25
priority: low
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

Current FineWeb classification uses fixed taxonomy (genre, author stance). Instead, adapt scoring axes per-analysis to whatever dimensions best explain the assistant axis structure.

E.g., if a subset of docs clusters by formality rather than genre, score on formality; if another clusters by audience level, score on that.

Let Claude propose the most discriminative axes after seeing the actual data distribution rather than forcing pre-defined categories.

Applies to: tail taxonomy, category projections, any future corpus classification.

Compute: minimal (classification prompt changes only, reuse existing projections).
