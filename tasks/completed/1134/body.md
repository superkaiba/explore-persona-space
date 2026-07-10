---
title: 'daily-fix: codify or ban the adopt-severe reconciler skip'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cff90f82864b
- daily-auto-filed
created_at: '2026-07-08T06:59:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): ccc66ab4 (#825, 09:45Z):
  interp-critic round 2 split Claude REVISE vs Codex PASS and the orchestrator adopted
  the more severe verdict WITHOUT spawning the reconciler ("the residual is mechanically
  verifiable, the fix is cheaper"), deviating from the documented ensemble-disagreement
  protocol.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

one sentence in the ensemble-disagreement step: either permit adopting the strictly-more-severe verdict without a reconciler when the residual is mechanically verifiable, or restate that the reconciler is mandatory

## Workflow gap

- **Bug observed:** ccc66ab4 (#825, 09:45Z): interp-critic round 2 split Claude REVISE vs Codex PASS and the orchestrator adopted the more severe verdict WITHOUT spawning the reconciler ("the residual is mechanically verifiable, the fix is cheaper"), deviating from the documented ensemble-disagreement protocol.
- **Why it is a workflow gap:** Protocol text and observed practice diverge; either the text or the practice should change so future sessions do not each re-litigate it.

## Proposed change

Disambiguate the ensemble-disagreement step (recovery matrix ~line 9378): either codify the shortcut (orchestrator MAY skip the reconciler by adopting the strictly-more-severe verdict when the flagged residual is mechanically verifiable and the fix is cheaper than a reconciler spawn) or restate the reconciler as mandatory. Today's deviation was conservative (extra revision, not skipped review).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: ccc66ab4 (#825) 09:45Z.
