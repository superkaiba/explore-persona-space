---
title: 'workflow-fix: make audit-claim HF tests hermetic (fail on pristine main)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6d7779b7da09
created_at: '2026-07-02T09:51:28Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation on #851 Step 9c: tests/test_verify_task_body_audit_claim.py::test_denial_and_hf_genuinely_missing_passes
  + ::test_hf_http_error_is_unverified_not_fail fail on pristine main because the
  audit-availability check finds the install_probes raw_completions.json now EXISTS
  on the HF data repo; live-HF-coupled workflow-invariant tests are flaky by construction.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #851 (Step 9c test-verdict gate, 2026-07-02).

## Goal

Make the audit-availability HF-Hub tests in `tests/test_verify_task_body_audit_claim.py` hermetic (mock the hub lookup) so they do not fail when live HF repo state changes.

## Workflow gap

- **Bug observed:** `tests/test_verify_task_body_audit_claim.py::test_denial_and_hf_genuinely_missing_passes` and `::test_hf_http_error_is_unverified_not_fail` fail on pristine `main` (verified 2026-07-02 at repo root, 2 failed in 1.96s): the audit-availability check finds `install_probes .../seed42/__no_system__/raw_completions.json` now EXISTS at `superkaiba1/explore-persona-space-data@a64f6fd7`, where the tests expect a genuinely-missing / http-error scenario.
- **Why it is a workflow gap:** workflow-invariant tests that depend on LIVE HF Hub state are flaky by construction — any upload to the data repo can flip them, and they then taint every issue's Step 9c touched-scope run (they sit in the select_step9c_tests workflow-invariant subset), forcing per-issue pre-existing-failure triage.
- **Confidence (emitter):** high (reproduced on pristine main twice: once in the #851 worktree run, once at repo root).

## Proposed change (candidate diff sketch — refine in planning)

```
+ In tests/test_verify_task_body_audit_claim.py, monkeypatch the HF-Hub
+ existence probe used by the audit-availability check (the helper in
+ scripts/verify_task_body.py that resolves repo files) so:
+   - test_denial_and_hf_genuinely_missing_passes fabricates a genuinely-missing
+     artifact (mock returns not-found) instead of relying on a real path being
+     absent from superkaiba1/explore-persona-space-data;
+   - test_hf_http_error_is_unverified_not_fail fabricates an HTTP error (mock
+     raises) instead of relying on live network behavior.
+ No behavior change to verify_task_body.py itself unless its probe seam needs
+ a small injection point.
```

## Scope / surfaces

- Primary target: `tests/test_verify_task_body_audit_claim.py`
- Possible secondary: the audit-claim probe seam in `scripts/verify_task_body.py` (injection point only, no behavior change).
- Grep the workflow surface for other live-HF-coupled tests while there (`grep -rln "explore-persona-space-data" tests/`).

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` passes; ruff on touched files passes.
- The check's PRODUCTION behavior (real HF lookups during real body verification) is unchanged — only the tests become hermetic.

## Provenance

- workflow_fix_target: tests/test_verify_task_body_audit_claim.py
- fingerprint: 6d7779b7da09
