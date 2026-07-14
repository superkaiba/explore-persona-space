---
title: 'daily-fix: verify_plan c15 misses pytest-backed fail-loud'
kind: infra
tags:
- wf-fix
- wf-fix-fp:09b3989de6c2
- daily-auto-filed
created_at: '2026-07-14T06:44:42Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-13 problem sweep (route 2): verify_plan.py c15_failloud_test_coverage
  WARN false-positives on plans whose fail-loud claim is backed by a named pytest.raises
  negative control - the vocabulary-adjacency heuristic missed it on #1296''s plan
  v1 AND v2'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-13 from the transcript problem sweep (session 384f17c7, task #1296).

## Goal

Stop `verify_plan.py` c15 (`c15_failloud_test_coverage`) from WARN-false-positiving on plans whose fail-loud claim is backed by a named `pytest.raises` negative control.

## Workflow gap

- **Bug observed:** on #1296's plan the c15 WARN persisted across v1 AND v2 even though the plan named a `pytest.raises(ModuleNotFoundError)` negative control for the fail-loud claim — the vocabulary-adjacency heuristic did not recognize it, and the session had to acknowledge-and-ship the WARN with a justification both rounds.
- **Why it is a workflow gap:** a recurring false-positive WARN trains sessions to rubber-stamp c15 acknowledgments, eroding the check's signal for the true-positive case it exists for.
- **Confidence (emitter):** medium (WARN-only; cost is noise + acknowledgment friction, not a wrong merge)
- verified-at-filing: `grep -n "c15" scripts/verify_plan.py` → check defined at :2038 (`c15_failloud_test_coverage`, "fail-loud acceptance claim backed by a test"), WARN-only conditional per the :43 table (2026-07-14 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Extend the c15 evidence scan to accept a `pytest.raises(` mention (with a named exception) in the plan's test/acceptance section as fail-loud test coverage, alongside the current vocabulary-adjacency patterns; add a regression fixture from #1296's plan v2 wording.

## Scope / surfaces

- Primary targets: `scripts/verify_plan.py`, `tests/test_verify_plan.py`

## Constraints / invariants

- WARN-only semantics unchanged (no new FAILs); replay the existing plan corpus to confirm no experiment-plan verdict flips (the #1291 replay harness pattern).
- Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 09b3989de6c2

Origin: transcript-mined (session 384f17c7, ~09:33Z and ~09:40Z — the WARN on plan v1 and v2 of #1296). Not a parked candidate — surfaced by the /daily problem sweep.
