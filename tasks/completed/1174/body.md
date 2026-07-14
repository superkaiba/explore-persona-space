---
title: 'workflow-fix: relative cache exclusion in improver-spawn lin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6695cda7b856
- daily-auto-filed
created_at: '2026-07-09T06:58:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): check_no_workflow_improver_spawn''s
  ''/.claude/cache/'' + ''/.claude/agent-memory/'' exclusion substring-matches ABSOLUTE
  paths (scripts/workflow_lint.py:5372), so under a repo-nested TMPDIR a tmp test
  repo rooted inside a real .claude/cache is wholesale-excluded and test_check_no_workflow_improver_spawn_flags_a_stray_spawn
  breaks.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1012 (recursion-guarded workflow-fix session).

## Goal

Make check_no_workflow_improver_spawn's cache/agent-memory exclusion relative-to-root so the check is hermetic under a repo-nested TMPDIR.

## Workflow gap

- **Bug observed:** check_no_workflow_improver_spawn's '/.claude/cache/' + '/.claude/agent-memory/' exclusion substring-matches ABSOLUTE paths (scripts/workflow_lint.py:5372), so under a repo-nested TMPDIR a tmp test repo rooted inside a real .claude/cache is wholesale-excluded and test_check_no_workflow_improver_spawn_flags_a_stray_spawn breaks.
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/workflow_lint.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
- s = p.as_posix()
- if "/.claude/cache/" in s or "/.claude/agent-memory/" in s:
+ rel = p.relative_to(root).as_posix()
+ if rel.startswith(".claude/cache/") or rel.startswith(".claude/agent-memory/"):
      continue
```

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- origin: parked candidate on task #1012 at 2026-07-05T04:34:11Z

Verbatim parked note:

> parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). Prose candidate from Claude code-reviewer r1: pre-existing test-hermeticity nit at scripts/workflow_lint.py:4640 — the '/.claude/cache/' exclusion substring-matches absolute paths, so test_check_no_workflow_improver_spawn_flags_a_stray_spawn breaks under a repo-nested TMPDIR; suggested fix: make the exclusion relative-to-root. Not routed by this session; surfaced for the next human/orchestrator pass.
