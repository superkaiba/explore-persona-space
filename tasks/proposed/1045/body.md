---
title: 'daily-fix: choom-protect Step 9c gates + VM long fits by def'
kind: infra
tags:
- wf-fix
- wf-fix-fp:366211557464
- daily-auto-filed
created_at: '2026-07-05T07:02:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): earlyoom SIGTERM''d Step
  9c pytest gates and long CPU fits repeatedly on 2026-07-04 under fleet memory pressure:
  #906''s full-suite gate killed twice, #995''s gate killed at ~42% (badness 969),
  #811''s re-fit killed ~2h in, and the 06:38 UTC storm killed #742 phase 4 + #813''s
  sweep pid. Each session improvised choom -n -600 protection after the kill.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Make choom -n -600 (or a systemd-run scope) the DEFAULT launch form for the Step 9c full-suite gate and for VM-side multi-hour fit launches (with per-cell checkpoints), instead of per-session improvisation after a kill.

## Workflow gap

- **Bug observed:** earlyoom SIGTERM'd Step 9c pytest gates and long CPU fits repeatedly on 2026-07-04 under fleet memory pressure: #906's full-suite gate killed twice, #995's gate killed at ~42% (badness 969), #811's re-fit killed ~2h in, and the 06:38 UTC storm killed #742 phase 4 + #813's sweep pid. Each session improvised choom -n -600 protection after the kill.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, .claude/rules/vectorize-many-cell-fits.md`
- Sessions: 4ea4c2b6 (#906, 15:13+ UTC), 10bacdc9 (#995, 21:33 UTC), 86923fc5 (#811, 03:57 UTC), 8149e2b5/d36ef80b (#742/#813, 06:38 UTC). Complementary: #1031 (plan-time RAM/RSS routing gate) covers plan-time budgeting; this task covers launch-time protection.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, .claude/rules/vectorize-many-cell-fits.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
