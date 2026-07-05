---
title: 'daily-fix: Step 9c known-red-on-main baseline ledger'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b1dcc59feb5b
- daily-auto-filed
created_at: '2026-07-04T21:36:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-02 backfill route-2: >=7 sessions re-derived pre-existing-ness
  of red main (2072+ ruff errors, failing tests)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-02 backfill problem sweep (route 2 —
behavior/logic change, independent review required). Emitting context:
transcript mining of the 2026-07-02 sessions.

## Goal

Maintain a known-red-on-main baseline ledger (failing tests + repo-wide ruff count at a main SHA) that the Step 9c test-verdict diffs against, so sessions stop re-deriving pre-existing-ness

## Workflow gap

- **Bug observed:** On 2026-07-02 at least 7 sessions independently burned a verification round proving red tests/lint were pre-existing on main (2072-2084 ruff errors, failing audit-claim and thread-caps tests)
- **Why it is a workflow gap:** The Step 9c verdict has no shared baseline of what is already red on main, so every code-path session pays a pristine-main cross-check round for failures its diff never caused
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ scripts/step9c_baseline.py refresh  # writes .claude/cache/
+   step9c-baseline.json {main_sha, failing_tests[], ruff_count}
+ select_step9c_tests / SKILL.md Step 9c: verdict compares the run's
+   failures against the ledger; only NEW failures block; stale ledger
+   (main advanced >N commits) => refresh first
```

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py, .claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every
  hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py, .claude/skills/issue/SKILL.md
- fingerprint: b1dcc59feb5b
- related_task: #852
- source: /daily 2026-07-02 backfill problem sweep (route 2)
