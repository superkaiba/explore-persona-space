---
title: 'daily-fix: lint check for bang-backtick in skill markdown'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8a3f57c9c04d
- daily-auto-filed
created_at: '2026-07-11T06:52:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): a code span ending in ''!''
  immediately before its closing backtick in a skill/agent markdown file is executed
  by the Claude Code dynamic-context preprocessor as an inline shell command; commit
  90af0ce2d9 (#1243) introduced two such spans in .claude/skills/issue/SKILL.md and
  EVERY /issue session spawned after ~14:30Z 2026-07-10 died at boot with zero assistant
  rows (>=8 dead sessions, tasks #1251-#1'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

add a workflow_lint check (e.g. --check-skill-bang-backtick) FAILing any non-dollar-preceded bang-immediately-before-backtick sequence in .claude/skills/**/*.md + .claude/agents/*.md, wired into the no-flags run so the Step 10d lint gate blocks the next occurrence

## Workflow gap

- **Bug observed:** a code span ending in '!' immediately before its closing backtick in a skill/agent markdown file is executed by the Claude Code dynamic-context preprocessor as an inline shell command; commit 90af0ce2d9 (#1243) introduced two such spans in .claude/skills/issue/SKILL.md and EVERY /issue session spawned after ~14:30Z 2026-07-10 died at boot with zero assistant rows (>=8 dead sessions, tasks #1251-#1256 stuck at proposed)
- **Provenance / evidence:** miner-01 P1 part 2 / miner-02 P2 optional hardening, /daily 2026-07-10 transcript sweep. The two live spans were hotfixed by the /daily run itself (commit f75e1b4c13); this lint prevents recurrence.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 8a3f57c9c04d

- workflow_fix_target: scripts/workflow_lint.py
