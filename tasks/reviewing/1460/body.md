---
title: 'daily-fix: lint inline free-analysis commits pre-push'
kind: infra
tags:
- wf-fix
- wf-fix-fp:76b100354436
- daily-auto-filed
created_at: '2026-07-17T06:57:40Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): inline free-analysis rounds
  commit scripts straight to main bypassing the code-review/lint pipeline — two bare
  .list_repo_tree( hub-verify offenders (issue1073:102, issue1092:92) landed that
  way and failed tests/test_workflow_lint.py at every Step 9c gate fleet-wide until
  #1388 fixed them'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1388/#1368/#1370/#1381 sessions each burned a pristine-main classification round on the two inline-landed offenders).

## Goal

Stop inline-round commits from landing lint-red scripts on main that every later session's gate must re-classify.

## Workflow gap

- **Bug observed:** inline free-analysis rounds commit scripts straight to main bypassing the code-review/lint pipeline — two bare .list_repo_tree( hub-verify offenders (issue1073:102, issue1092:92) landed that way and failed tests/test_workflow_lint.py at every Step 9c gate fleet-wide until #1388 fixed them
- **Why it is a workflow gap:** The inline carve-out deliberately skips the pipeline for speed; a cheap lint step is the minimum defense for repo-wide invariants.
- **Confidence (emitter):** medium-high (4 sessions burned rounds this cycle)
- verified-at-filing: `grep -n 'Same-turn completion contract' CLAUDE.md` -> the inline contract names commit+push+fold but no lint step (absence claim; SKILL.md 9a-ter mirror likewise); offenders since fixed by #1388 (waiver comments now at issue1073:102 / issue1092:92 — the ROOT-CAUSE gap remains)

## Proposed change (candidate diff sketch — refine in planning)

add a lint step to the inline free-analysis same-turn completion contract: run workflow_lint (at least the hub-verify + invariant-relevant checks) on the round's committed scripts before the push

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 76b100354436

