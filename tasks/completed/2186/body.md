---
title: 'workflow-fix: Split the pin across the two post-compaction sites — assert '
kind: infra
tags:
- wf-fix
- wf-fix-fp:8ca2da863105
- urgent-main-red
created_at: '2026-08-08T00:03:26Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: tests/test_suffixed_pod_completion_teardown_pin.py\n\
  bug_observed: test_claude_md_carries_completion_side_teardown_contract fails on\
  \ origin/main — it requires \"Completion-side teardown\" >= 2 times in CLAUDE.md\
  \ (both #1662 sites), but the 2026-08-05/06 compaction moved the § Pods multi-pod\
  \ paragraph site into .claude/rules/pods.md, leaving exactly 1 occurrence in CLAUDE.md\
  \ (the inline-override carve-out) and 1 in pods.md.\nwhy_workflow_gap: The compaction\
  \ train landed without re-running the workflow-invariant pin family on the final\
  \ merged tree, leaving a stale count-based pin that every intervening Step 9c gate\
  \ must re-classify as known-red.\nproposed_change: Split the pin across the two\
  \ post-compaction sites — assert 1 occurrence (with its load-bearing tokens) in\
  \ CLAUDE.md's inline-override carve-out AND 1 in .claude/rules/pods.md's suffixed-pod\
  \ paragraph — instead of counting 2 in CLAUDE.md.\ndiff_sketch: |\n  -     assert\
  \ _norm(body).count(\"Completion-side teardown\") >= 2, (\n  +     assert _norm(body).count(\"\
  Completion-side teardown\") >= 1, (...)\n  +     pods_rule = (REPO / \".claude\"\
  \ / \"rules\" / \"pods.md\").read_text()\n  +     assert \"Completion-side teardown\"\
  \ in _norm(pods_rule)\nconfidence: high\nrelated_task: #2157\nurgency: main-red\n\
  failing_test: tests/test_suffixed_pod_completion_teardown_pin.py::test_claude_md_carries_completion_side_teardown_contract\n\
  wf_fix: true\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#2157. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_suffixed_pod_completion_teardown_pin.py::test_claude_md_carries_completion_side_teardown_contract` red on origin/main: Split the pin across the two post-compaction sites — assert 1 occurrence (with its load-bearing tokens) in CLAUDE.md's inline-override carve-out AND 1 in .claude/rules/pods.md's suffixed-pod paragraph — instead of counting 2 in CLAUDE.md.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** test_claude_md_carries_completion_side_teardown_contract fails on origin/main — it requires "Completion-side teardown" >= 2 times in CLAUDE.md (both #1662 sites), but the 2026-08-05/06 compaction moved the § Pods multi-pod paragraph site into .claude/rules/pods.md, leaving exactly 1 occurrence in CLAUDE.md (the inline-override carve-out) and 1 in pods.md.
- **Failing node (router-verified):** `tests/test_suffixed_pod_completion_teardown_pin.py::test_claude_md_carries_completion_side_teardown_contract`
- **Confidence (emitter):** high
- verified-at-filing: step9c ledger @ 2026-08-07T08:10:45Z lists tests/test_suffixed_pod_completion_teardown_pin.py::test_claude_md_carries_completion_side_teardown_contract red-on-main

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `tests/test_suffixed_pod_completion_teardown_pin.py`
- Failing node: `tests/test_suffixed_pod_completion_teardown_pin.py::test_claude_md_carries_completion_side_teardown_contract`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- This session carries a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route its own subagents' workflow-fix candidates (recursion
  guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: tests/test_suffixed_pod_completion_teardown_pin.py
- fingerprint: 8ca2da863105
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_suffixed_pod_completion_teardown_pin.py
bug_observed: test_claude_md_carries_completion_side_teardown_contract fails on origin/main — it requires "Completion-side teardown" >= 2 times in CLAUDE.md (both #1662 sites), but the 2026-08-05/06 compaction moved the § Pods multi-pod paragraph site into .claude/rules/pods.md, leaving exactly 1 occurrence in CLAUDE.md (the inline-override carve-out) and 1 in pods.md.
why_workflow_gap: The compaction train landed without re-running the workflow-invariant pin family on the final merged tree, leaving a stale count-based pin that every intervening Step 9c gate must re-classify as known-red.
proposed_change: Split the pin across the two post-compaction sites — assert 1 occurrence (with its load-bearing tokens) in CLAUDE.md's inline-override carve-out AND 1 in .claude/rules/pods.md's suffixed-pod paragraph — instead of counting 2 in CLAUDE.md.
diff_sketch: |
  -     assert _norm(body).count("Completion-side teardown") >= 2, (
  +     assert _norm(body).count("Completion-side teardown") >= 1, (...)
  +     pods_rule = (REPO / ".claude" / "rules" / "pods.md").read_text()
  +     assert "Completion-side teardown" in _norm(pods_rule)
confidence: high
related_task: #2157
urgency: main-red
failing_test: tests/test_suffixed_pod_completion_teardown_pin.py::test_claude_md_carries_completion_side_teardown_contract
wf_fix: true
<!-- /workflow-fix-candidate -->
