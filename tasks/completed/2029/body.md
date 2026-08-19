---
title: 'workflow-fix: verify_plan edited-literal pin-test check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2732a7f36293
- daily-auto-filed
created_at: '2026-08-03T06:39:53Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): plan v2 for #1948 edited
  a SKILL.md probe-pattern literal without listing the pin test (tests/test_issue_skill_gate_single_flight.py:106)
  that asserts that exact literal — acceptance criterion "Step 9c compare exit 0"
  was deterministically unsatisfiable as scoped; caught only by the Claude critic.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-08-02 Step C parked-candidate routing pass from a
workflow-fix candidate parked on task #1948 (emitting agent: the #1948 round-1
Claude critic, Methodology lens, via the orchestrator's recursion-guard park;
`epm:workflow-fix-candidate` at 2026-08-03T00:39:29Z).

## Goal

Add a verify_plan.py check: when a plan declares an edit to a SKILL.md /
workflow-surface literal, every tests/ file pinning that literal must appear in
the plan's edit-target list (WARN/FAIL per unlisted pin-test hit).

## Workflow gap

- **Bug observed:** plan v2 for #1948 edited a SKILL.md probe-pattern literal without listing the pin test (tests/test_issue_skill_gate_single_flight.py:106) that asserts that exact literal — acceptance criterion "Step 9c compare exit 0" was deterministically unsatisfiable as scoped; caught only by the Claude critic.
- **Why it is a workflow gap:** no mechanical check requires that when a plan declares an edit to a SKILL.md/workflow-surface literal, every tests/ file pinning that literal appears in the plan's edit-target list.
- **Confidence (emitter):** low
- verified-at-filing: `grep -n -iE 'pin.test|edited.literal|literal.*tests/|check_edited' scripts/verify_plan.py` → 24 hits in 1 file, all in the DURABILITY-PIN arm (a plan must DECLARE a `Durability pin:` for new prose and the pin must ship in the diff — verify_plan.py:5246–5443, #1679 arm at :6405) — none implements the inverse direction this candidate proposes (detect EXISTING tests/ pins on a plan-EDITED literal and require them in the edit-target list), so the gap claim binds; plus `git log --oneline --since='7 days ago' -- scripts/verify_plan.py` → 7 commits, none touching edited-literal pin-test detection (2026-08-02). The incident's pin test exists: tests/test_issue_skill_gate_single_flight.py:106 asserts the exact step9c probe literal (read at compose time).

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_edited_literal_pin_tests(plan_text, ...):
+     for lit in plan-declared edited literals (fenced old->new pairs on workflow-surface files):
+         hits = grep -rln <lit> tests/
+         WARN/FAIL per hit file not named in the plan's File-paths section
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Related but distinct open sibling on the same file: #1960 (lattice parser
  negated-existence atom, fp 63fb274bb974, status reviewing) — different bug,
  different fingerprint; not a duplicate.
- Grep the workflow surface for the pattern before editing and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 2732a7f36293

Origin candidate (verbatim, parked on #1948 at 2026-08-03T00:39:29Z):

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_plan.py
bug_observed: plan v2 for #1948 edited a SKILL.md probe-pattern literal without listing the pin test (tests/test_issue_skill_gate_single_flight.py:106) that asserts that exact literal — acceptance criterion "Step 9c compare exit 0" was deterministically unsatisfiable as scoped; caught only by the Claude critic.
why_workflow_gap: no mechanical check requires that when a plan declares an edit to a SKILL.md/workflow-surface literal, every tests/ file pinning that literal appears in the plan's edit-target list.
proposed_change: add a verify_plan.py check — for each plan-declared edited literal/pattern on a workflow-surface file, grep tests/ for the literal and FAIL/WARN if a hit file is absent from the plan's target list.
confidence: low
related_task: #1948
<!-- /workflow-fix-candidate -->
