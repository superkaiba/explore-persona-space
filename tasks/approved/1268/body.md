---
title: 'daily-fix: Step 10d repin/guard snippet hardening bundle'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f9f2c9a6b2c8
- daily-auto-filed
created_at: '2026-07-11T06:52:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): three recurring Step 10d
  recovery gaps each re-derived in-session on 2026-07-10: (1) the certification repin
  loop runs checkout <MAIN_SHA> -- <path> over foreign tasks/ paths and fails ''pathspec
  did not match'' on paths main has MOVED/DELETED (task folders move on status change
  - #1242 13:37Z, #1246 14:43Z); (2) the overlay-listing diff quotes non-ASCII payload
  filenames via quotePath and silently'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

fold into the Step 10d snippets: (1) on_main/gone_on_main discrimination in the repin loop (checkout the former, git rm --cached / drop the latter); (2) -c core.quotePath=false on the overlay-listing diff; (3) one re-fetch + re-pin retry in Guard 1 on a mid-guard main advance

## Workflow gap

- **Bug observed:** three recurring Step 10d recovery gaps each re-derived in-session on 2026-07-10: (1) the certification repin loop runs checkout <MAIN_SHA> -- <path> over foreign tasks/ paths and fails 'pathspec did not match' on paths main has MOVED/DELETED (task folders move on status change - #1242 13:37Z, #1246 14:43Z); (2) the overlay-listing diff quotes non-ASCII payload filenames via quotePath and silently skips them (#1212 code-reviewer Minor); (3) Guard 1's churn-pinned strip abandons when origin/main advances mid-guard instead of re-fetching + re-pinning once (#1224 ~09:23Z)
- **Provenance / evidence:** miner-03 P7 + P11.1 and miner-01 P2 residual, /daily 2026-07-10 transcript sweep. Grouped: all three are Step 10d certification/guard snippet edits in the same file.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: f9f2c9a6b2c8

- workflow_fix_target: .claude/skills/issue/SKILL.md
