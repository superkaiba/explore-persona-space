---
title: 'daily-fix: verbatim bare push/merge snippets in SKILL.md'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6cb2c14b7dde
- daily-auto-filed
created_at: '2026-07-08T07:00:06Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): The piped-git guard hook
  blocked ~10 piped push/merge attempts across >=8 sessions on 2026-07-07 (c86ff35c
  x3, efacefad, 250c1e88, 16126574, 05000ba2, 9cc76509, 983af21b, 1e8cb3e4, 12d6ae88,
  ee574f2e); c86ff35c re-composed the same piped shape immediately after its first
  block. Each block costs a wasted turn.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

put the bare-form push / merge command snippets verbatim into the /issue Step 5 / Step 9b / Step 10d command templates so sessions copy instead of composing piped variants

## Workflow gap

- **Bug observed:** The piped-git guard hook blocked ~10 piped push/merge attempts across >=8 sessions on 2026-07-07 (c86ff35c x3, efacefad, 250c1e88, 16126574, 05000ba2, 9cc76509, 983af21b, 1e8cb3e4, 12d6ae88, ee574f2e); c86ff35c re-composed the same piped shape immediately after its first block. Each block costs a wasted turn.
- **Why it is a workflow gap:** The hook contains the failure but the recurring attempts show the templates induce composing-with-pipes; verbatim snippets remove the composition step.

## Proposed change

Add the exact bare-form snippets (push with exit-code check; worktree-scoped bare merge) verbatim at the SKILL.md points where sessions currently compose piped variants (Step 5 commit/push, Step 9b merge, Step 10d merge), with a one-line never-pipe reminder adjacent.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: guard-hook blocks enumerated across sweep batches 1/3/5/6/7 of 2026-07-07.
