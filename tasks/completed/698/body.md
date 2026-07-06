---
title: 'workflow-fix: add gate-id-uniqueness check to workflow_lint.py'
kind: infra
tags:
- wf-fix
- wf-fix-fp:573498928966
created_at: '2026-06-28T06:40:24Z'
has_clean_result: false
parent_id: 694
origin_prompt: 'Claude code-reviewer r1 follow-up on task #694: workflow_lint.py is
  missing a gate-id-uniqueness check across workflow.yaml gates.{inline, park_and_wait,
  conditional}; the duplicate id=15 in round 1 slipped past existing lint because
  nothing validates id uniqueness.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a Claude r1 code-reviewer follow-up surfaced on task #694 (emitting agent: code-reviewer).

## Goal

Add a gate-id-uniqueness check to `scripts/workflow_lint.py` so that a duplicate `id:` value across `gates.{inline, park_and_wait, conditional}` in `.claude/workflow.yaml` is caught mechanically rather than relying on a human / Claude code-reviewer noticing it.

## Workflow gap

- **Bug observed:** task #694 round 1 added a new `related_work_positioning` conditional gate with `id: 15`, which was already held by the existing `concern_deferral_request` gate (line 403). Both `workflow_lint.py --default` and `--check-asks` PASSed (the `id:` field is never read for gate resolution, so functionally nothing broke); the duplicate id only surfaced via the Claude `code-reviewer`'s manual inspection.
- **Why it is a workflow gap:** the `id:` field is conventionally unique-per-gate-list across the workflow.yaml `gates.*` blocks (the existing ids 1-16 are all unique pre-#694), and the next future addition of a `gates.{inline, park_and_wait, conditional}` entry SHOULD be checked mechanically against reusing an existing id. Letting a duplicate id slip past lint into `main` again is a known recurring failure mode (Claude's verdict: "this slipped past every existing gate precisely because no check validates `id`").
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

Add a check function to `scripts/workflow_lint.py` (next to the existing `--check-asks` / `--check-tables` / `--check-references` checks) that:
1. Parses `.claude/workflow.yaml`.
2. Collects ALL `id:` values across `gates.inline`, `gates.park_and_wait`, and `gates.conditional`.
3. FAILs with a clear error if any id appears more than once, naming the offending gate-name pairs.
4. Bundles into the no-flags default run (so `uv run python scripts/workflow_lint.py` catches it without a flag).

Also: add a `tests/test_workflow_lint.py::test_workflow_lint_check_gate_ids_unique_exits_zero` test (mirroring the existing `test_workflow_lint_check_asks_exits_zero` shape) to pin the check.

## Scope / surfaces

- Primary targets: `scripts/workflow_lint.py` + `tests/test_workflow_lint.py`
- Grep before editing to understand the existing check structure:
  `grep -n 'def check_' scripts/workflow_lint.py`

## Constraints / invariants

- Workflow-helper script + test — `scripts/workflow_lint.py` is in-scope per workflow-fix-on-bug.md § Workflow surface.
- `uv run ruff check .` on touched files PASSes; existing `workflow_lint.py --default + --check-asks` baseline PASSes.
- The new check is bundled into the default run (no separate flag required to invoke).
- Must run cleanly on the current `workflow.yaml` after task #694's gate id renumber (id 15 → 17 for `related_work_positioning`); the existing ids 1-14, 16, 17 are all unique.

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: 573498928966

(Surfaced as Claude `code-reviewer` r1 Minor finding + follow-up suggestion on task #694 — verbatim: "I also surfaced a follow-up suggesting a `workflow_lint.py` gate-id-uniqueness check, since this slipped past every existing gate precisely because no check validates `id`.")
