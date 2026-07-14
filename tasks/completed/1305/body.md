---
title: 'daily-fix: implementer pre-report 9c touched-scope run'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c2850d74e9f8
- daily-auto-filed
created_at: '2026-07-14T06:44:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-13 problem sweep (route 2): the implementer''s self-chosen
  local verification scope is narrower than the Step 9c gate''s touched-scope selection,
  so a changed pinned literal passed local verification but failed the gate round
  1 (#1288, ~30 min rework)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-13 from the transcript problem sweep (session 5cbde72b, task #1288 — the only Step 9c gate FAIL round of the day's 15-task infra drain).

## Goal

Make the implementer's pre-report verification scope match the Step 9c gate's touched-scope selection, so a diff that changes a pinned literal cannot pass local verification and then fail the gate.

## Workflow gap

- **Bug observed:** #1288's implementer round 1 changed a pinned literal in the Step 10d command sequence but did not update the 3 tests pinning it in `tests/test_step10d_guard3.py`; its self-chosen local verification (14 pin/regression tests, all green) missed them, while the 9c touched-scope selection (37 files) ran them → gate round-1 FAIL, one full implementer + reviewer + gate rework round (~30 min).
- **Why it is a workflow gap:** the implementer spec never tells the implementer to run (or approximate) the same `select_step9c_tests.py --base main` selection the gate will run — the verification-scope mismatch is structural, not a one-off oversight.
- **Confidence (emitter):** medium-high (single clean incident, structural cause)
- verified-at-filing: `grep -n "select_step9c_tests" .claude/agents/implementer.md .claude/agents/experiment-implementer.md` → 0 hits in both (2026-07-14 UTC) — the pre-report selection duty is absent from both implementer specs.

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/agents/implementer.md` (and the Step 4b brief template in `.claude/skills/issue/SKILL.md`, plus `experiment-implementer.md` if the planner finds the same gap there):

```
+ Before reporting success, run the SAME touched-scope test selection the
+ Step 9c gate will run: `uv run python scripts/select_step9c_tests.py --base main`
+ from the worktree, and run the selected files — or at minimum every test
+ file that greps any literal/symbol the diff changed. A local verification
+ scope narrower than the gate's selection is the #1288 rework shape.
```

## Scope / surfaces

- Primary targets: `.claude/agents/implementer.md`, `.claude/skills/issue/SKILL.md` (Step 4b brief)
- Secondary: `.claude/agents/experiment-implementer.md`

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` default run passes.
- Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: .claude/agents/implementer.md, .claude/skills/issue/SKILL.md
- fingerprint: c2850d74e9f8

Origin: transcript-mined (session 5cbde72b, ~08:00-08:30Z; task #1288 events `epm:test-verdict` round-1 FAIL → round-2 PASS). Not a parked candidate — surfaced by the /daily problem sweep.
