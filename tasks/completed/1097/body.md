---
title: 'daily-fix: Step 10d lint verdict consumed before gh pr merge'
kind: infra
tags:
- wf-fix
- wf-fix-fp:72ecdd823040
- daily-auto-filed
created_at: '2026-07-07T06:49:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-06 problem sweep (route 2): The Step 10d merge template
  rm -f''s the consume-once lint-verdict file BEFORE `gh pr merge` runs; when the
  merge fails for non-lint transport reasons (''can''t be rebased'' / merge conflicts)
  the PASS verdict is orphaned and the retry has no gate. #1082''s session hand-recreated
  the gate file (`echo pass > /tmp/issue-1082-lint-verdict.txt`) — a self-attested
  gate write, exactly what consume-once was m'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-06 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Consume (rm) the lint-verdict file only AFTER the `gh pr merge` attempt returns success; invalidate the verdict on new lint-surface commits instead of on read, so a rebase-refusal→squash retry stays certified by the still-valid gate run rather than a hand-written verdict.

## Workflow gap

- **Bug observed:** The Step 10d merge template rm -f's the consume-once lint-verdict file BEFORE `gh pr merge` runs; when the merge fails for non-lint transport reasons ('can't be rebased' / merge conflicts) the PASS verdict is orphaned and the retry has no gate. #1082's session hand-recreated the gate file (`echo pass > /tmp/issue-1082-lint-verdict.txt`) — a self-attested gate write, exactly what consume-once was meant to prevent. Hit twice on 2026-07-06: sessions 24257d06 (#1081, PR #842) and d7ae9a3b (#1082, PR #843), both the documented #1041 rebase-refusal→squash shape.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- source: /daily 2026-07-06 problem sweep (transcript-mined)
