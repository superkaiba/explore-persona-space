---
title: 'daily-fix: detached runs declare a harvest step at launch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:74bdf2d31ee0
- daily-auto-filed
created_at: '2026-07-24T06:48:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): the detached long-compute
  recipe has no default harvest contract so results collection assumes the launching
  session survives'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). Incident (#1310 session): a long detached VM run was launched, then the user asked "if I close this terminal will it continue running?" — the run itself was detached (safe), but the HARVEST (collect results, commit, fold into the body) was session-bound; an autoharvest step was bolted on only after the user raised it.

## Goal

Make session-independent harvest the DEFAULT for detached long-compute phases: the detached-launch recipe in `/issue` SKILL.md gains a standing "harvest step" requirement — a detached results-collection/commit step (or a durable breadcrumb the next tick/session harvests from) declared AT LAUNCH, so a closed session never strands finished results.

## Workflow gap

- **Bug observed:** the SKILL.md detached VM-side long-compute recipe covers pid+log breadcrumbs and completion tracking, but has no default harvest contract — results collection assumes the launching session survives. `grep -n "harvest" .claude/skills/issue/SKILL.md` → only the batch-judge self-harvest lines (714/722); no detached-phase harvest step (absence claim, in-target).
- **Why it is a workflow gap:** detached phases exist precisely because sessions die/close; leaving harvest session-bound reintroduces the single point of failure the detachment removed.
- **Confidence:** medium
- verified-at-filing: grep above (2026-07-24 UTC).

## Proposed change (refine in planning)

Extend SKILL.md § Detached VM-side long compute phases: the launch declares its harvest path (a detached post-run collection step writing to the durable artifact locations + a completion sentinel the tick triage reads), mirroring the batch-judge deadline-bounded self-harvest pattern.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants

- Workflow-surface only; recursion guard applies (`workflow_fix_target:` Provenance line).

## Provenance

- fingerprint: 74bdf2d31ee0

- workflow_fix_target: .claude/skills/issue/SKILL.md
