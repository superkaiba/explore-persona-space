---
title: 'workflow-fix: Raise-by-default _hf_tree_get guard in verify-'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3cf194735d13
- daily-auto-filed
created_at: '2026-07-09T06:57:51Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): tests/test_verify_task_body.py
  lacks the raise-by-default _hf_tree_get autouse guard adopted in #860''s tests/test_verify_task_body_audit_claim.py
  — a probe test that forgets to stub _hf_tree_get relies on the suite-wide EPM_VERIFY_BODY_NO_HF
  fence (skip-not-raise semantics), so a missed mock passes silently as ''unverified''
  instead of failing loud.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #860 by a recursion-guarded workflow-fix session.

## Goal

Add a small autouse fixture inheriting the _no_unexpected_probes pattern: default _hf_tree_get to a raiser; tests that intend probe traffic use _stub_tree as today — giving the sibling file the same network-independent missed-mock detection.

## Workflow gap

- **Bug observed:** tests/test_verify_task_body.py lacks the raise-by-default _hf_tree_get autouse guard adopted in #860's tests/test_verify_task_body_audit_claim.py — a probe test that forgets to stub _hf_tree_get relies on the suite-wide EPM_VERIFY_BODY_NO_HF fence (skip-not-raise semantics), so a missed mock passes silently as 'unverified' instead of failing loud.
- **Why it is a workflow gap:** Missed-mock probe tests silently degrade to skip semantics, hiding coverage loss in the HF-pin verification checks; the sibling file already demonstrated the guard shape.
- **Confidence (emitter):** low-medium
- **Sweep verification (2026-07-08):** Verified 2026-07-08: the only autouse fixture in test_verify_task_body.py (_clear_hf_existence_cache, line 1668) clears memo caches; _stub_tree (line 1688) is opt-in per test; no raise-by-default guard exists. The conftest EPM_VERIFY_BODY_NO_HF fence gives network independence but skip-not-raise semantics — the candidate's missed-mock-detection gap is real. Small additive test-only change.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; add `@pytest.fixture(autouse=True) def _no_unexpected_probes(monkeypatch): monkeypatch.setattr(verify_task_body, '_hf_tree_get', _raise_on_call)` mirroring test_verify_task_body_audit_claim.py, ordered so _stub_tree overrides it)

## Scope / surfaces

- Primary target: `tests/test_verify_task_body.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: tests/test_verify_task_body.py
- origin: parked candidate on task #860 at 2026-07-02T11:11:15Z

Verbatim parked note:

> parked — running under workflow_fix_target recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard). source: prose-followup (Claude code-reviewer r1, Minor standing rec). target_file: tests/test_verify_task_body.py. proposed_change: the sibling test file lacks the raise-by-default _hf_tree_get autouse guard adopted in #860's tests/test_verify_task_body_audit_claim.py — inheriting the _no_unexpected_probes pattern would give its probe tests the same network-independent missed-mock detection. confidence: low-medium (sibling already has an autouse cache-clear + _stub_tree; the guard is a small additive fixture). routed: parked for the next non-workflow-fix orchestrator pass / nightly sweep.
