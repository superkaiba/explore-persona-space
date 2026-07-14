---
title: 'daily-fix: producer-side regen bullet in upload-policy'
kind: infra
tags:
- wf-fix
- wf-fix-fp:da06c89cbf61
- daily-auto-filed
created_at: '2026-07-13T06:44:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): #779 regenerated published
  question artifacts IN PLACE (HF commit 9578892ef4) with no version bump and no note
  naming dependent captures, silently invalidating #922''s day-older cx.pt capture
  — the producer-side half of the #922 pair-incoherence incident.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 Step C parked-candidate routing pass, from the formal candidate parked on task #941 (2026-07-03T22:10:39Z, recursion guard; emitting agent: #941's round-1 Alternatives critic).

## Goal

Add a producer-side bullet to the Upload Policy — regenerating a published artifact in place requires either a version-bumped path or an explicit note naming known dependent captures.

## Workflow gap

- **Bug observed:** #779 regenerated published question artifacts IN PLACE (HF commit 9578892ef4) with no version bump and no note naming dependent captures, silently invalidating #922's day-older cx.pt capture — the producer-side half of the #922 pair-incoherence incident.
- **Why it is a workflow gap:** the Upload Policy governs artifact publication but has no rule about regenerating an already-published artifact that other tasks' captures depend on; #941's consumer-side (j) check detects the incoherence, but nothing discourages producing it.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "Regenerating a published artifact" .claude/rules/upload-policy.md` → 0 hits (the bullet is still absent) (2026-07-13).

## Proposed change (candidate diff sketch — refine in planning)

```diff
+ **Regenerating a published artifact in place:** when re-uploading /
+ reconstructing an artifact other tasks may have captured under, either
+ (a) publish at a version-bumped path (issueN_<slug>/v2/...), or
+ (b) record a regeneration note naming known dependent captures, so the
+ item-(j) pairwise provenance check has provenance to read (#922/#779).
```

## Scope / surfaces

- Primary target: `.claude/rules/upload-policy.md`
- Sibling: `.claude/rules/artifact-reuse.md` item (j) (the consumer-side check this bullet feeds).

## Constraints / invariants

- Workflow-surface only. Lint gates pass. Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: .claude/rules/upload-policy.md
- fingerprint: da06c89cbf61

Verbatim origin candidate: formal `<!-- workflow-fix-candidate v1 -->` block parked on #941 events.jsonl at 2026-07-03T22:10:39Z (fp da06c89cbf61, related_task #941).
