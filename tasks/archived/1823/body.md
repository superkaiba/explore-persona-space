---
title: 'workflow-fix: re-wrap verify_task_body.py line 13224 (the inclusive-cap f-'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1178a640171e
- urgent-main-red
created_at: '2026-07-29T07:57:31Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/verify_task_body.py\n\
  bug_observed: tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset\
  \ fails on a clean origin/main tree — scripts/verify_task_body.py:13224:101: E501\
  \ Line too long (106 > 100), introduced by main commit c341f3bd59 (2026-07-28 route-1\
  \ inclusive-cap message); bare ruff check passes because pyproject per-file-ignores\
  \ relax scripts/*, exactly the #1672 masking mechanism.\nwhy_workflow_gap: the full-ruleset\
  \ policy pin on live workflow helpers is red fleet-wide, so every intervening session's\
  \ Step 9c gate must re-classify the red (the #1643/#1681 per-hour cost class) until\
  \ the one-line fix lands.\nproposed_change: re-wrap verify_task_body.py line 13224\
  \ (the inclusive-cap f-string fragment) to <=100 chars so the policy pin is green\
  \ on main.\ndiff_sketch: |\n  -                f\"<{V3_FINDING_PROSE_FAIL_WORDS};\
  \ FAIL fires at ≥{V3_FINDING_PROSE_FAIL_WORDS} inclusive)\"\n  +               \
  \ f\"<{V3_FINDING_PROSE_FAIL_WORDS}; FAIL fires at \"\n  +                f\"≥{V3_FINDING_PROSE_FAIL_WORDS}\
  \ inclusive)\"\nconfidence: high\nrelated_task: #1786\nurgency: main-red\nfailing_test:\
  \ tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset\n\
  wf_fix: true\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1786. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset` red on origin/main: re-wrap verify_task_body.py line 13224 (the inclusive-cap f-string fragment) to <=100 chars so the policy pin is green on main.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset fails on a clean origin/main tree — scripts/verify_task_body.py:13224:101: E501 Line too long (106 > 100), introduced by main commit c341f3bd59 (2026-07-28 route-1 inclusive-cap message); bare ruff check passes because pyproject per-file-ignores relax scripts/*, exactly the #1672 masking mechanism.
- **Failing node (router-verified):** `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset -q` -> rc=1 at main @ f14606eeae (2026-07-29T07:57:15Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Failing node: `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- This session carries a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route its own subagents' workflow-fix candidates (recursion
  guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 1178a640171e
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
bug_observed: tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset fails on a clean origin/main tree — scripts/verify_task_body.py:13224:101: E501 Line too long (106 > 100), introduced by main commit c341f3bd59 (2026-07-28 route-1 inclusive-cap message); bare ruff check passes because pyproject per-file-ignores relax scripts/*, exactly the #1672 masking mechanism.
why_workflow_gap: the full-ruleset policy pin on live workflow helpers is red fleet-wide, so every intervening session's Step 9c gate must re-classify the red (the #1643/#1681 per-hour cost class) until the one-line fix lands.
proposed_change: re-wrap verify_task_body.py line 13224 (the inclusive-cap f-string fragment) to <=100 chars so the policy pin is green on main.
diff_sketch: |
  -                f"<{V3_FINDING_PROSE_FAIL_WORDS}; FAIL fires at ≥{V3_FINDING_PROSE_FAIL_WORDS} inclusive)"
  +                f"<{V3_FINDING_PROSE_FAIL_WORDS}; FAIL fires at "
  +                f"≥{V3_FINDING_PROSE_FAIL_WORDS} inclusive)"
confidence: high
related_task: #1786
urgency: main-red
failing_test: tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset
wf_fix: true
<!-- /workflow-fix-candidate -->
