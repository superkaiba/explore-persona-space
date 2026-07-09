---
title: 'workflow-fix: Widen thread-caps invariant test target globs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6f182ba61982
- daily-auto-filed
created_at: '2026-07-09T06:59:37Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The shared-VM thread-caps
  invariant test scans only scripts/issue*_*.py + src/explore_persona_space/experiments/**/run_*.py
  (tests/test_shared_vm_thread_caps.py:547-548, 601-602), leaving scripts/analyze_results.py-class
  entrypoints and non-run_*.py src modules outside the invariant.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1146.

## Goal

Extend the shared-VM thread-caps invariant test's coverage to the entrypoint classes its current globs miss, with an offender-fix round for the newly covered files.

## Workflow gap

- **Bug observed:** The shared-VM thread-caps invariant test scans only scripts/issue*_*.py + src/explore_persona_space/experiments/**/run_*.py (tests/test_shared_vm_thread_caps.py:547-548, 601-602), leaving scripts/analyze_results.py-class entrypoints and non-run_*.py src modules outside the invariant.
- **Why it is a workflow gap:** the failure originates in the workflow surface named below, not in any one experiment.
- **Confidence (emitter):** see parked note

## Proposed change (candidate diff sketch — refine in planning)

  - targets = sorted(root.glob("scripts/issue*_*.py")) + sorted(
  -     root.glob("src/explore_persona_space/experiments/**/run_*.py"))
  + targets = ... + sorted(root.glob("scripts/analyze_results.py-class entrypoints"))
  +           + non-run_*.py src experiment modules (enumerate + offender round;
  +             exact glob set is the planning decision)

## Scope / surfaces

- Primary target: `tests/test_shared_vm_thread_caps.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: tests/test_shared_vm_thread_caps.py
- origin: parked candidate on task #1146 at 2026-07-08T14:52:07Z

Verbatim parked note:

```
parked — running under workflow_fix_target Provenance (recursion guard, workflow-fix-on-bug.md § Recursion guard). Prose follow-up surfaced by Alternatives critic r1: widening the invariant test's TARGET GLOB set (scripts/issue*_*.py + src/experiments/**/run_*.py) to cover scripts/analyze_results.py-class entrypoints + non-run_*.py src modules is a distinct predicate-vs-target-set task with its own offender round. target_file: tests/test_shared_vm_thread_caps.py. Logged for the nightly /daily parked-candidate routing pass; NOT auto-routed from this session.
```
