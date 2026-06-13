---
title: 'WildChat-category contexts for leakage prediction: cluster WildChat, pick
  2 separable categories, sample realistic user completions'
kind: experiment
tags: []
created_at: '2026-06-12T19:49:10Z'
has_clean_result: false
origin_prompt: 'task #616 cross-check: consolidated from docs/mentor_updates/2026-06-11.md'
goal: Determine whether clustered, separable WildChat conversation categories work
  as realistic training/eval contexts for the context-leakage predictor.
---
## Goal

Determine whether clustered, separable WildChat conversation categories work as realistic training/eval contexts for the context-leakage predictor.


## Summary

Build WildChat-derived context categories for the context-leakage prediction eval:

1. Cluster normal WildChat conversations into categories.
2. Pick 2 categories that separate cleanly (verify separability, e.g. by activation geometry or a probe).
3. Use them as training/eval contexts in the leakage-prediction grid.
4. Where WildChat prefixes are used as contexts, consider sampling realistic user completions onto them instead of using raw prefixes alone.

## Motivation

The long-context predictor already works on WildChat prefixes (layer-21 cosine rho = 0.89), but the context battery so far leans on persona prompts and synthetic rewraps. Naturally occurring conversation categories are the realistic next tier of contexts (data-realism preference order: real-world data first).

## Scope

Capture only. Fold into the comprehensive context-leakage eval when picked up; cross-reference the behavior-generalization testbed (#545) and the context-structure result (#594).

## Provenance

Consolidated from the 2026-06-11 mentor meeting notes (docs/mentor_updates/2026-06-11.md) during the task #616 cross-check.
