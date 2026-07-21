---
title: 'workflow-fix: inline_lint_gate conservative-blocks on pytest warnings-summary
  node-id headers'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c1ea5912e6c7
created_at: '2026-07-21T07:00:23Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation, #1112 rankem inline round: inline_lint_gate
  reproducibly BLOCKs a green payload test file because pytest''s warnings summary
  emits the node id (path::test, no lineno) as a header line; scanner branch scripts/inline_lint_gate.py:352;
  forced a recorded EPM_ALLOW_ROOT_CODE_COMMIT=1 override on commit 6321bc5e40.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the orchestrator's own observation during the #1112 rankem inline round (user-chat inline override).

## Goal

Teach `inline_lint_gate.py`'s payload-naming scanner to classify pytest warnings-summary node-id header lines (`path::test_name`, no lineno, inside the warnings summary section) as WARN/report instead of a conservative BLOCK.

## Workflow gap

- **Bug observed:** the gate reproducibly BLOCKs a 29/29-green payload test file: pytest's warnings summary attributes environmental DeprecationWarnings (torch/swig, `<frozen importlib._bootstrap>`) to the FIRST test that triggered them, emitting a header line that is exactly the node id `tests/test_issue1112_rankem_m5_overflow.py::test_overflow_callback_uploads_and_prunes` — a payload-naming line with no parseable lineno — and the scanner's conservative branch (scripts/inline_lint_gate.py:352) BLOCKs on it. Reproduced twice (gate logs /tmp/issue-1112-inline-gate5.log + gate6.log, 2026-07-21); the same file certified cleanly in an earlier run whose warnings attribution differed. The block forced the sanctioned `EPM_ALLOW_ROOT_CODE_COMMIT=1` override (recorded, commit 6321bc5e40) for a legitimately green payload.
- **Why it is a workflow gap:** the inline payload lint gate is the commit-time guard for every direct-to-main code payload; a reproducible false BLOCK on any payload whose (slow) tests trigger environmental warnings routes legitimate commits through the emergency override, eroding the gate's authority and training sessions to reach for the override.
- **Confidence (emitter):** high (twice-reproduced; scanner branch identified at scripts/inline_lint_gate.py:352).
- verified-at-filing: `grep -n 'parseable lineno' scripts/inline_lint_gate.py` → conservative-block branch at :352 (plus contract comment :26) (2026-07-21); reproduction: gate5 + gate6 logs both carry `payload-naming hit without a parseable lineno (conservative block): tests/test_issue1112_rankem_m5_overflow.py::test_overflow_callback_uploads_and_prunes` while `uv run pytest tests/test_issue1112_rankem_m5_overflow.py tests/test_issue1112_rankem_dispatch.py -q` reads 29 passed; `git log --oneline --since='7 days ago' -- scripts/inline_lint_gate.py` → 1 commit (8248be9501, the gate's own landing, #1500/#1272) — no fix for this class landed.

## Proposed change (candidate diff sketch — refine in planning)

diff_sketch: |
  In the payload-naming scan, before the conservative-block branch (:352):
  + track whether the scanner is inside pytest's "warnings summary" section
  +   (header line "=+ warnings summary =+" .. next "=+" section header);
  + a payload-naming line inside that section matching the bare node-id shape
  +   `^\S+::\w+$` (no lineno, no FAILED/ERROR token) is a WARN/report line
  +   ("warnings-summary attribution"), never a conservative BLOCK;
  keep the conservative block for naming hits outside recognized sections.

## Scope / surfaces

- Primary target: `scripts/inline_lint_gate.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'parseable lineno' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The gate's conservative posture for genuinely unattributable failure lines is preserved — only the warnings-summary node-id header class is reclassified.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/inline_lint_gate.py
- fingerprint: c1ea5912e6c7

Verbatim surfaced observation (orchestrator, #1112 rankem inline round, 2026-07-21): gate 5 returned INCONCLUSIVE (script) + conservative BLOCK (test file, warnings-summary node-id header); gate 6 re-run certified the script and reproduced the BLOCK on the identical header line; suite green 29/29; commit 6321bc5e40 shipped under the recorded EPM_ALLOW_ROOT_CODE_COMMIT=1 override.
