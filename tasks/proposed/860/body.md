---
title: 'workflow-fix: pre-existing HF audit-claim test failures on main'
kind: infra
tags:
- wf-fix
- wf-fix-fp:19092e467e3c
created_at: '2026-07-02T10:01:15Z'
has_clean_result: false
origin_prompt: 'Auto-filed from #852 Step 9c: tests/test_verify_task_body_audit_claim.py::test_denial_and_hf_genuinely_missing_passes
  + ::test_hf_http_error_is_unverified_not_fail fail on current main (pre-existing,
  AssertionError line 229); fix verify_task_body.py or the test''s HF error mocking.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #852 (emitting agent: orchestrator, during the Step 9c test-verdict gate).

## Goal

Fix the two pre-existing failing HF audit-claim tests on main (test_denial_and_hf_genuinely_missing_passes, test_hf_http_error_is_unverified_not_fail).

## Workflow gap

- **Bug observed:** `tests/test_verify_task_body_audit_claim.py::test_denial_and_hf_genuinely_missing_passes` and `::test_hf_http_error_is_unverified_not_fail` fail on current main with AssertionError at line 229, independent of any branch (confirmed by a repo-root control run during #852's Step 9c gate, 2026-07-02: `2 failed, 2 passed`).
- **Why it is a workflow gap:** `scripts/verify_task_body.py` is a workflow-surface verifier and these tests pin its HF audit-claim behavior (genuinely-missing → PASS; HF HTTP error → UNVERIFIED not FAIL); a red invariant test on main poisons every issue branch's Step 9c touched-scope gate (it fired inside #852's gate and had to be triaged as pre-existing).
- **Confidence (emitter):** medium (root cause not yet diagnosed — either a verify_task_body.py regression or HF Hub API/error-shape drift; both tests assert False at test line 229).

## Proposed change (candidate diff sketch — refine in planning)

```
# Diagnose first: run the two tests with -x --tb=long on main.
# Likely either:
+ scripts/verify_task_body.py: restore the genuinely-missing/HTTP-error classification
+   (UNVERIFIED, never FAIL, on HF HTTP errors; PASS on denial+genuinely-missing), OR
+ tests/test_verify_task_body_audit_claim.py: update the mocked HF error shape to the
+   current huggingface_hub exception contract if the library drifted.
```

## Scope / surfaces

- Primary target: `tests/test_verify_task_body_audit_claim.py`, `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing (`grep -rn 'audit_claim\|genuinely_missing' scripts/ tests/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: tests/test_verify_task_body_audit_claim.py, scripts/verify_task_body.py
- fingerprint: 19092e467e3c

Surfaced prose (verbatim): during #852's Step 9c gate, 4 subset failures triaged — 2 branch-caused (fixed in fc52c1ed94), 2 PRE-EXISTING on main: tests/test_verify_task_body_audit_claim.py::test_denial_and_hf_genuinely_missing_passes + ::test_hf_http_error_is_unverified_not_fail (AssertionError line 229, control run at repo root on main).
