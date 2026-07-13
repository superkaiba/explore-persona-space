---
title: 'daily-fix: source-pin the piped-git-push bundling test'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3512ee8f9016
- daily-auto-filed
created_at: '2026-07-13T06:44:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): test_workflow_lint_piped_git_push_bundled_in_no_flags
  (tests/test_workflow_lint.py:2411) pins no-flags bundling via exit-0-on-a-clean-tree
  — the identical vacuous class #1233 just fixed for pipe-python.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 Step C parked-candidate routing pass, from the formal candidate parked on task #1233 (2026-07-10T10:44:13Z, recursion guard; emitting agent: #1233's code-reviewer bug-class sweep).

## Goal

Replace with a source-dispatch pin (regex on the dispatch branch at scripts/workflow_lint.py + tuple-membership substring), the same exemplar-2 pattern #1233 landed.

## Workflow gap

- **Bug observed:** test_workflow_lint_piped_git_push_bundled_in_no_flags (tests/test_workflow_lint.py:2411) pins no-flags bundling via exit-0-on-a-clean-tree — the identical vacuous class #1233 just fixed for pipe-python.
- **Why it is a workflow gap:** The bundled-vs-opt-in distinction is unpinned for the piped-git-push check; an un-bundling regression would pass this test (same #712 §4f class).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "bundled_in_no_flags" tests/test_workflow_lint.py` → the vacuous form confirmed still present at :2412 (`result = _run(); assert result.returncode == 0`), while source-pin variants exist for pipe_python (:2123) and push_failure_swallow (:2629) but NOT piped-git-push (2026-07-13).

## Proposed change (candidate diff sketch — refine in planning)

```diff
- def test_workflow_lint_piped_git_push_bundled_in_no_flags(): result = _run(); assert result.returncode == 0
+ def test_piped_git_push_bundled_in_no_flags_source_pin():
+     src = _LINT.read_text(encoding="utf-8")
+     assert re.search(r"if args\.check_piped_git_push or no_flags:\s*\n\s*errors\.extend\(check_piped_git_push\(\)\)", src)
+     assert "or args.check_piped_git_push" in src
```

## Scope / surfaces

- Primary target: `tests/test_workflow_lint.py`
- Reference implementation: #1233's exemplar-2 source-pin pattern (test_pipe_python_bundled_in_no_flags_source_pin, :2123).

## Constraints / invariants

- Workflow-surface only. Lint + ruff pass. Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: tests/test_workflow_lint.py
- fingerprint: 3512ee8f9016

Verbatim origin candidate: formal `<!-- workflow-fix-candidate v1 -->` block parked on #1233 events.jsonl at 2026-07-10T10:44:13Z (fp 3512ee8f9016, related_task #1233).
