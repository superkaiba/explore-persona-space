---
title: 'daily-fix: register 3 unregistered step9c pin tests'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3c4eb1c23efa
- daily-auto-filed
created_at: '2026-07-24T06:46:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): tests test_ownership_probe_exemplar_bracketed,
  test_daily_three_route_classifier_doc, test_daily_stub_first_doc are registered
  in neither WORKFLOW_INVARIANT nor the manifest so they never run on later diffs
  of their pinned files'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-23 parked-candidate routing pass (Step C). Raised as recursion-guard-parked candidates on tasks #1627 (planner, 13:36Z) and #1630 (planner 15:05Z; code-reviewer 15:33Z widened the enumeration).

## Goal

Register the three existing-but-unregistered workflow-pin tests in `scripts/select_step9c_tests.py` WORKFLOW_INVARIANT + `tests/step9c_workflow_invariant_manifest.txt` so they actually run on later diffs of the files they pin (the #1546 unregistered-pin class).

## Workflow gap

- **Bug observed:** three existing pin tests are registered in NEITHER `WORKFLOW_INVARIANT` nor the manifest, so none ever runs via the Step 9c selector on a later diff of its pinned file (`.md` files have no selector discovery arm): (1) `tests/test_ownership_probe_exemplar_bracketed.py` (#1495; pins SKILL.md/CLAUDE.md ownership-probe text), (2) `tests/test_daily_three_route_classifier_doc.py`, (3) `tests/test_daily_stub_first_doc.py` (both pin `.claude/skills/daily/SKILL.md` prose).
- **Why it is a workflow gap:** an unregistered SKILL.md pin test is dead weight — the selector's literal/stem arms are .py/.sh-only by design, so registration is the only execution path.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c "test_ownership_probe_exemplar_bracketed\|test_daily_three_route_classifier_doc\|test_daily_stub_first_doc" scripts/select_step9c_tests.py tests/step9c_workflow_invariant_manifest.txt` → 0 hits in BOTH named targets (absence-of-registration claim: 0-hit in-target IS the evidence); `ls tests/test_ownership_probe_exemplar_bracketed.py tests/test_daily_three_route_classifier_doc.py tests/test_daily_stub_first_doc.py` → all three test files exist (2026-07-24 UTC). Open task #865 on the same file is a DIFFERENT bug (selector worktree-blindness), not a dedup hit.

## Proposed change (candidate diff sketch — refine in planning)

Per test: one tuple entry in `WORKFLOW_INVARIANT` + one sorted manifest line (the #1593 two-line registration recipe), pinning each test to the workflow file(s) it guards.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`, `tests/step9c_workflow_invariant_manifest.txt`

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 3c4eb1c23efa

- workflow_fix_target: scripts/select_step9c_tests.py, tests/step9c_workflow_invariant_manifest.txt

Origin: parked candidates on #1627 (2026-07-23T13:36:33Z) and #1630 (15:05:03Z + 15:33:39Z widened block: "TWO existing daily-skill prose pins are unregistered in WORKFLOW_INVARIANT — tests/test_daily_three_route_classifier_doc.py AND tests/test_daily_stub_first_doc.py").
