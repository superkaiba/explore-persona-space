---
title: 'daily-fix: /daily route-1 code applies run full-ruleset lint'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d7b950fc9026
- daily-auto-filed
created_at: '2026-07-30T07:02:24Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): The 07-28 /daily route-1
  commit c341f3bd59 shipped an E501 to main that bare ruff missed (per-file-ignores
  masking) while tests/test_ruff_policy.py''s full-ruleset check caught it — main
  stayed red 40+h and every session''s Step 9c gate re-paid classification'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miners F-P1, A-P2, J-P1(c) — root-cause of the fleet-wide 07-29 main-red).

## Goal

A /daily route-1 self-applied code fix must not ship lint-red to main; the verification gate must run the same ruleset the fleet's gate tests enforce.

## Workflow gap

- **Bug observed:** Commit c341f3bd59 (07-28 route-1) introduced a 106-char line in scripts/verify_task_body.py; bare `ruff check` passes it (per-file-ignores mask E501 for that file) while tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset runs ruff under a stricter config and failed — red on main ~40h, >=10 sessions re-paid gate classification (~20-39 min each), duplicate filing #1843 archived.
- **Why it is a workflow gap:** the /daily SKILL's 'Verification gate for code fixes' names syntax/import/test checks but not the full-ruleset lint the fleet gate enforces; the guard_root_code_commit hook now requires the inline lint gate for direct-to-main code, but that gate's mapped tests did not include test_ruff_policy for this payload.
- **Confidence (emitter):** medium
- verified-at-filing: `uv run pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset -q` -> failed at 06:35Z, passed after tonight's re-wrap (commit in tonight's daily); `git log --oneline -2 -- scripts/verify_task_body.py` -> c341f3bd59 is the introducing commit (2026-07-30, this run).

## Proposed change (refine in planning)

Add to the route-1 verification gate: for touched *.py under scripts/, run the full-ruleset ruff invocation test_ruff_policy uses (or simply run that test node) before commit; note the per-file-ignores masking trap explicitly.

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/daily/SKILL.md
- fingerprint: d7b950fc9026
