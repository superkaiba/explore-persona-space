---
title: 'workflow-fix: green test_ruff_policy full-ruleset — verify_task_body.py:13224
  E501'
kind: infra
tags:
- wf-fix
- 'wf-fix-fp:'
created_at: '2026-07-30T03:17:55Z'
has_clean_result: false
origin_prompt: 'implementer-surfaced candidate on #1738 sae-arm round: main-red test_ruff_policy
  E501 at verify_task_body.py:13224 (c341f3bd59)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1738 (emitting agent: experiment-implementer, sae-arm round).

## Goal

Fix the pre-existing E501 (line >100 chars) in scripts/verify_task_body.py line 13224 so tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset is green on main.

## Workflow gap

- **Bug observed:** tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset is RED on origin/main — scripts/verify_task_body.py:13224 is 109 chars (E501 under the test's full ruleset), introduced by commit c341f3bd59 (daily 2026-07-28 route-1: inclusive-cap message edit).
- **Why it is a workflow gap:** a red workflow-invariant test on pristine main forces every intervening session's Step 9c gate + Step 10d TG legs to re-classify it as pre-existing (the #1643/#1713 per-hour fleet cost class).
- **Confidence (emitter):** high
- verified-at-filing: `awk 'NR==13224' scripts/verify_task_body.py | wc -c` → 109 chars; `uv run pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset -q` on main → 1 failed (rc=1, AssertionError at tests/test_ruff_policy.py:129); default-ruleset `ruff check --select E501` passes (the repo default excludes E501 — the test's FULL ruleset is what fails) (2026-07-30)

## Proposed change (candidate diff sketch — refine in planning)

Wrap/shorten the 109-char line at scripts/verify_task_body.py:13224 (message string from the c341f3bd59 inclusive-cap edit) to <=100 chars; re-run the named test node to green.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for other >100-char lines the same commit introduced before editing; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 and carries a workflow_fix_target: Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: (computed by filer tags)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
bug_observed: tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset red on origin/main — verify_task_body.py:13224 is 109 chars (E501 under the test's full ruleset), from commit c341f3bd59
why_workflow_gap: red workflow-invariant test on pristine main taxes every session's Step 9c / TG gate with re-classification
proposed_change: shorten/wrap verify_task_body.py:13224 to <=100 chars; test node green
diff_sketch: |
  - <the 109-char message line>
  + <same message wrapped across two lines / shortened>
confidence: high
related_task: #1738
urgency: main-red
failing_test: tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset
wf_fix: true
<!-- /workflow-fix-candidate -->
