---
title: 'daily-fix: widen lazy-import pin scan set (from #1304)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:57c387f2695f
- daily-auto-filed
created_at: '2026-07-15T06:51:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): the widened scripts-local
  lazy-import pin test scans only scripts/backend_poll.py + scripts/pod_config.py;
  the same unguarded-lazy-import class survives in any OTHER module-mode-imported
  script'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 Step C parked-candidate routing pass from a candidate parked under the recursion guard on task #1304 (emitting agent: Alternatives critic on #1304's plan review; park ts 2026-07-14T07:43:50Z).

## Goal

widen the scripts-local lazy-import pin scan set to all scripts/ files imported in module mode by src//tests/ (or all scripts/*.py with a lazy-import site), guarding newly-flagged sites in the same round

## Workflow gap

- **Bug observed:** the widened scripts-local lazy-import pin test scans only scripts/backend_poll.py + scripts/pod_config.py; the same unguarded-lazy-import class survives in any OTHER module-mode-imported script
- **Why it is a workflow gap:** the module-name axis was generalized by #1304 but the file axis remains fixed, so a future scripts-local lazy import in a third module-mode-imported script (e.g. pod_lifecycle.py, spawn_session.py) is uncovered.
- **Confidence (emitter):** low-medium (scope deliberately excluded from #1304 per its Goal + kill criterion)
- verified-at-filing: `grep -n '"backend_poll.py", "pod_config.py"' tests/test_backend_poll.py` -> 1 hit at :3222 (per-target: tests/test_backend_poll.py = 1 hit; the scan loop iterates exactly those two filenames) (2026-07-15). Retraction re-check on #1304 events after the park ts: none (task completed + merged with the file-axis scope still excluded).

## Proposed change (refine in planning)

Widen the pin test's scan set from the hardcoded two-file tuple to every scripts/*.py imported in module mode by src/ or tests/ (or every scripts/*.py containing a lazy-import site); guard any newly-flagged site in the same round.

## Scope / surfaces

- Primary target: `tests/test_backend_poll.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a workflow_fix_target Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: tests/test_backend_poll.py
- fingerprint: 57c387f2695f

Verbatim parked candidate (prose park on #1304, 2026-07-14T07:43:50Z):

> target_file: tests/test_backend_poll.py
> bug_observed: the widened scripts-local lazy-import pin test scans only scripts/backend_poll.py + scripts/pod_config.py; the same unguarded-lazy-import class survives in any OTHER module-mode-imported script (e.g. pod_lifecycle.py, spawn_session.py).
> why_workflow_gap: file-scoped pin -- the module-name axis was generalized (#1304) but the file axis remains fixed.
> proposed_change: widen the scan set to all scripts/ files imported in module mode by src//tests/ (or all scripts/*.py with a lazy-import site), guarding any newly-flagged sites in the same round.
> confidence: low-medium
> related_task: #1304
