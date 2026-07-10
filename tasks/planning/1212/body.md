---
title: 'daily-fix: Step 10d lint gate runs origin/main lint vintage'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ab28662f7498
- daily-auto-filed
created_at: '2026-07-09T07:01:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): #1112 Step 10d lint gate
  false-blocked twice (stale ratchet caps; worktree scripts/ tree predating a referenced
  helper), each forcing a fresh ~12-min SHA-bound gate re-run — ~35-45 min churn.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Stop Step 10d lint-gate false-blocks from lint-version / tree-vintage drift on long-lived worktree branches.

## Workflow gap

- **Bug observed:** Transcript 9e877609 (issue-1112) 19:15:49Z: stale ratchet caps flagged main-synced agent specs (main caps admit all five); 19:20:51Z: daily/SKILL.md reference check false-blocked because the worktree scripts/ tree predates the helper — file main-identical, merge-inert. 3 pre-merge gate runs + 1 post-merge run, ~35-45 min.
- **Why it is a workflow gap:** The gate prose runs the WORKTREE vintage of the linter + scripts tree against main-synced content; nothing tells the session to freshness-sync the gate tooling first.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

Step 10d gate block: before the first run, `git -C "$WT" checkout origin/main -- scripts/workflow_lint.py <referenced helpers>` (or run `git show origin/main:scripts/workflow_lint.py` via a temp copy); document the two observed drift classes.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-1112-1090 P4 (9e877609 19:15:49Z, 19:20:51Z)
