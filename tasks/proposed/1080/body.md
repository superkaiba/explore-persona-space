---
title: 'daily-fix: inline pod runs on terminal-status parents must p'
kind: infra
tags:
- wf-fix
- wf-fix-fp:182d531e61ea
- daily-auto-filed
created_at: '2026-07-06T06:58:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-05 problem sweep (route 2): The watcher pod-safety
  pass repeatedly auto-stopped pod-779 under a LIVE user-requested inline GPU follow-up
  (#779, parent parked at awaiting_promotion): the inline path had posted no epm:run-launched
  and no keep-running tag, so the watcher''s live-follow-up inference had nothing
  to key on. Bootstrap attempt 1 died mid-clone leaving .git/index.lock + a half-cloned
  repo; the session misdiagnosed a f'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-05 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Add one sentence to the CLAUDE.md 'User-chat inline free analysis' routing bullet and the SKILL.md 9a-ter block: any inline/user-chat run that provisions or reuses a pod on a terminal-status parent MUST post epm:run-launched AND task.py add-tag <N> keep-running BEFORE launch (the #477 watcher exemption keys on exactly these signals).

## Workflow gap

- **Bug observed:** The watcher pod-safety pass repeatedly auto-stopped pod-779 under a LIVE user-requested inline GPU follow-up (#779, parent parked at awaiting_promotion): the inline path had posted no epm:run-launched and no keep-running tag, so the watcher's live-follow-up inference had nothing to key on. Bootstrap attempt 1 died mid-clone leaving .git/index.lock + a half-cloned repo; the session misdiagnosed a flaky host, terminated + reprovisioned (~20 min + a fresh provision wasted) before identifying the watcher at 04:33Z. Session 5664c4f8, 2026-07-06.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `CLAUDE.md, .claude/skills/issue/SKILL.md`
- Doc-level but changes agent behavior -> route 2 for independent review.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: CLAUDE.md, .claude/skills/issue/SKILL.md
- source: /daily 2026-07-05 problem sweep (transcript-mined)
