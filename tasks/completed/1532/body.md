---
title: 'workflow-fix: restore ruff full-ruleset helper pin to green'
kind: infra
tags:
- wf-fix
- wf-fix-fp:447bfe6341a8
- daily-auto-filed
created_at: '2026-07-19T07:06:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset
  is red on pristine main with 8 errors (C901 x5, RUF001/2/3 x3) across the 4 pinned
  helpers, so the #1023 pin reads as pre-existing noise on every branch (6 sessions
  each burned a 0-introduced triage pass). Distinct scope from on_hold #1486 (repo-config
  residual).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
group raised on tasks #1524 + #1523 (formal candidate blocks, implementer
round 1) and independently surfaced as prose on #1498 (item 1), #1511
(item 1), #1516 (prose follow-up), and #1520 — six same-bug sightings merged
into ONE filing by the 2026-07-18 /daily Step C parked-candidate sweep
(primary candidate: #1524's, the latest and most complete).

## Goal

Restore `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset`
(the #1023 full-ruleset cleanliness pin on live workflow helpers) to green —
extract/simplify the over-cap C901 functions (or add annotated per-function
noqa with reasons, per the test's own "fix the helper" guidance) and replace
the ambiguous unicode minus/dash characters with ASCII.

## Workflow gap

- **Bug observed:** The durability pin
  `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset`
  FAILs on pristine origin/main with 8 errors in the 4 pinned helpers
  (C901 x4-5 over-cap functions — `_process_orphan_wrapper` 24>15,
  `_failover_gcp_to_runpod` 17>15, `select_tests_with_reasons` 16>15,
  `_gather_hf_count_claims` / `_gather_hf_unpinned_count_claims` 16>15 —
  plus RUF001/RUF002/RUF003 ambiguous-unicode hits in backend_poll.py and
  verify_task_body.py).
- **Why it is a workflow gap:** The #1023 full-ruleset cleanliness pin on
  live workflow helpers is permanently red on main, so it can no longer catch
  NEW regressions — any diff-linked run of this test reads as pre-existing
  noise, and at least four separate sessions (#1498, #1511, #1516, #1520,
  #1523, #1524) each burned a triage pass proving 0-introduced.
- **Confidence (emitter):** high
- verified-at-filing: `uv run python -m pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset -x -q` → 1 failed, "Found 8 errors" — red reproduced on main at compose time; per-target hits from the test's own ruff output: scripts/autonomous_session_watch.py 1 (C901 `_process_orphan_wrapper` L19844), scripts/backend_poll.py 3 (RUF002 L3092, C901 `_failover_gcp_to_runpod` L3456, RUF003 L3573), scripts/select_step9c_tests.py 1 (C901 `select_tests_with_reasons` L805), scripts/verify_task_body.py 3 (RUF001 L9182, C901 L9309, C901 L10377) — line numbers drifted slightly vs the candidates' enumeration (e.g. 9158→9182), same 8 findings. Dedup vs open on_hold #1486 ("ruff residual burn-down: ~33 errors + 17 format-dirty files"): #1486's scope is the DIFFERENT repo-config residual (16 src/explore_persona_space + 11 eps/experiments F401 + frozen-script F821s + format-dirty files) and does NOT name these 4 helpers or the full-ruleset pin — not a duplicate, but the spawned session should cross-reference it (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/backend_poll.py: MINUS SIGN -> ASCII '-' at L3092 docstring + L3573 comment (RUF002/RUF003)
scripts/verify_task_body.py: EN DASH -> ASCII '-' at ~L9182 (RUF001)
+ per-function extraction of a cohesive sub-block in each C901 offender
  (the _fold_cards extraction in #1524's f0dd46ca02 is the worked pattern),
  or noqa: C901 with reason where extraction is genuinely riskier.
Run the test to confirm `Found 0 errors`.
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py, scripts/backend_poll.py, scripts/select_step9c_tests.py, scripts/verify_task_body.py`
- Re-run the test's exact ruff invocation before editing (line numbers drift
  as main moves); if an offender is deliberately frozen, move it out of the
  pin list with a comment per the test's own instruction. Cross-reference
  on_hold #1486 (repo-config ruff residual) — adjacent but disjoint scope.

## Constraints / invariants

- Workflow-surface only — the 4 helpers are in-scope workflow-helper scripts.
- Behavior-preserving refactors only (these are live fleet automations:
  watcher, poller, gate selector, body verifier) — full pytest on their
  pinning tests after each extraction; a noqa-with-reason is preferred over a
  risky extraction.
- `scripts/workflow_lint.py` no-flags run + the ruff policy test pass at the end.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, scripts/backend_poll.py, scripts/select_step9c_tests.py, scripts/verify_task_body.py
- fingerprint: a75150a37d1c

<!-- workflow-fix-candidate v1 -->
target_file: scripts/autonomous_session_watch.py, scripts/backend_poll.py, scripts/select_step9c_tests.py, scripts/verify_task_body.py
bug_observed: The durability pin tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset FAILs on pristine origin/main with 8 errors in these 4 pinned helpers (5 C901 over-cap functions — _process_orphan_wrapper 24>15, _failover_gcp_to_runpod 17>15, select_tests_with_reasons 16>15, _gather_hf_count_claims 16>15, _gather_hf_unpinned_count_claims 16>15 — plus RUF001/RUF002/RUF003 ambiguous-unicode hits in backend_poll.py and verify_task_body.py).
why_workflow_gap: The #1023 full-ruleset cleanliness pin on live workflow helpers is permanently red on main, so it can no longer catch NEW regressions (any diff-linked run of this test reads as pre-existing noise — it cost this round a triage pass to prove 0-introduced).
proposed_change: Restore the pin to green — extract/simplify the 5 over-cap functions (or add annotated per-function noqa with reasons, per the test's own "fix the helper" guidance) and replace the 3 ambiguous unicode minus/dash characters with ASCII.
diff_sketch: |
  scripts/backend_poll.py: MINUS SIGN -> ASCII '-' at L3092 docstring + L3573 comment (RUF002/RUF003)
  scripts/verify_task_body.py: EN DASH -> ASCII '-' at L9158 (RUF001)
  + per-function extraction of a cohesive sub-block in each of the 5 C901 offenders
    (the _fold_cards extraction in this round's f0dd46ca02 is the worked pattern),
    or noqa: C901 with reason where extraction is genuinely riskier.
confidence: high
related_task: #1524
<!-- /workflow-fix-candidate -->

Merged same-bug sightings (deduped into this filing): #1523 candidate-block
(fingerprint 9c10ac278222, same 4-file target set, same 8 errors); #1498
prose item 1; #1511 prose item 1; #1516 prose follow-up; #1520 prose
(backend_poll.py + verify_task_body.py subset, attributing the drift to the
#1490/#1505/#1518 merges postdating the pin).
