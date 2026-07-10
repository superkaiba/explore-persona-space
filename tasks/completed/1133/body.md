---
title: 'daily-fix: dose-ladder checkpoint retention sizing rule'
kind: infra
tags:
- wf-fix
- wf-fix-fp:872452228cbf
- daily-auto-filed
created_at: '2026-07-08T06:59:15Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): #1112 (05000ba2): full-FT
  checkpoints ~15-28GB x 30 rungs filled the ft-7b default 200GB volume -> ENOSPC
  at rung 24/30, crash + terminate + fresh 800GB pod while billing $16/hr.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

add a between-rung checkpoint-retention clause to plan-compute-sizing.md: dose-to-band ladders keep only the dose-selected + latest checkpoints, cleaning non-selected rungs between rungs; plans size disk to the RETAINED set

## Workflow gap

- **Bug observed:** #1112 (05000ba2): full-FT checkpoints ~15-28GB x 30 rungs filled the ft-7b default 200GB volume -> ENOSPC at rung 24/30, crash + terminate + fresh 800GB pod while billing $16/hr.
- **Why it is a workflow gap:** #1118 fixed volume-size threading (provisioning side); the plan-time retention rule is the missing complement that keeps footprints bounded regardless of volume size.

## Proposed change

Add to .claude/rules/plan-compute-sizing.md (disk sizing): a dose-ladder / multi-rung training plan states its checkpoint-retention policy (keep dose-selected + latest only; clean non-selected rungs between rungs) and sizes disk to the RETAINED set; the critic REVISEs a ladder plan whose disk estimate assumes keeping every rung without justification. Upload-before-delete policy unchanged.

## Scope / surfaces

- Primary target: `.claude/rules/plan-compute-sizing.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: 05000ba2 (#1112) ENOSPC at rung 24/30; PM session e3f63b58 tracked the burn.
