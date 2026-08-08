---
title: 'workflow-fix: inline_lint_gate false-blocks payload on PASSING-test captured
  tracebacks'
kind: infra
tags:
- wf-fix
created_at: '2026-08-02T18:38:26Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1345 successor session: inline_lint_gate.py
  conservative block fired on captured stderr of PASSING designed-crash tests (572
  passed / 0 failed run); commit shipped under EPM_ALLOW_ROOT_CODE_COMMIT=1 escape,
  recorded on #1345 v242'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1345 (emitting agent: orchestrator, successor session 2026-08-02).

## Goal

Stop `scripts/inline_lint_gate.py` from conservatively BLOCKing a payload file on
traceback lines that appear in the captured output of PASSING tests (the pytest
`-rA` `PASSES` section), the sibling class of the #1585 warnings-summary fix.

## Workflow gap

- **Bug observed:** the gate's pytest leg BLOCKed `scripts/issue825_fit_cells.py`
  ("payload-naming hit without a parseable lineno (conservative block)") on two
  traceback frames (`File ".../issue825_fit_cells.py", line 2342, in
  _fit_within_cells` / `line 2269, in _apply_gates`) that were captured stderr of
  `tests/test_issue825_no_internal_gates.py::test_apply_gates_internal_crash_defers_with_flag`
  — a designed-crash test that PASSED; the run summary was 572 passed / 0 failed.
  The commit had to ship under the documented `EPM_ALLOW_ROOT_CODE_COMMIT=1`
  escape (recorded on #1345 events v242).
- **Why it is a workflow gap:** `evaluate()` treats ANY pytest-leg line containing
  the payload path as a payload-naming hit unless it is a warnings-summary
  attribution row (#1585) or carries a NON_RED_PREFIX; captured-output blocks of
  PASSING tests (`==== PASSES ====` fenced section, and `Captured stdout/stderr
  call` blocks of tests the summary lists as PASSED) are a report class the scan
  does not exclude, so any payload file exercised by a designed-crash /
  fail-loud-path test false-blocks whenever it is edited.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'payload-naming\|parseable lineno\|PASSES' scripts/inline_lint_gate.py` → the conservative-block branch at lines 440-453 and the #1585 `warnings_attribution_idxs` carve-out at lines 375-394 (no PASSES-section handling present); incident evidence in /tmp/issue-1345-inline-lint.txt (2026-08-02 18:29Z run: `=== PASSES ===` section carries the two traceback frames; terminal summary `572 passed`) (2026-08-02)

## Proposed change (candidate diff sketch — refine in planning)

```
+ PASSES_SECTION_TITLE_RE = re.compile(...)   # "==== PASSES ====" fence title
+ def passing_capture_idxs(pytest_lines): ... # same fenced-section tracker as
+     # warnings_attribution_idxs (#1585), keyed on the PASSES title; optionally
+     # also reclassify hits when PYTEST_SUMMARY_RE reports 0 failed/0 errors
  ws_idxs = warnings_attribution_idxs(pytest_lines)
+ pass_idxs = passing_capture_idxs(pytest_lines)
  combined = ... (ln, i in ws_idxs or i in pass_idxs) ...
```

Keep the conservative block verbatim for hits outside the PASSES section on runs
with a non-zero failed/error count.

## Scope / surfaces

- Primary target: `scripts/inline_lint_gate.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'payload-naming' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Pin with a fails-pre-fix test: a pytest-leg fixture whose PASSES section carries
  a payload-naming traceback frame with 0 failed must certify, and the same frame
  in a FAILED test's section must still block.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/inline_lint_gate.py
- fingerprint: (computed at filing by the wrapper's manifest tag)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/inline_lint_gate.py
bug_observed: the pytest-leg payload-naming scan conservatively BLOCKs a payload file on traceback lines captured from PASSING tests (pytest -rA PASSES section), despite a 572-passed/0-failed summary
why_workflow_gap: evaluate() excludes only warnings-summary rows (#1585) and NON_RED_PREFIX lines, so passing-test captured output naming the payload path is misread as red evidence
proposed_change: reclassify PASSES-section (and 0-failed-run) captured-output hits as report-class via the same fenced-section tracker as warnings_attribution_idxs
diff_sketch: |
  + pass_idxs = passing_capture_idxs(pytest_lines)  # PASSES-fence tracker, #1585 pattern
  + combined = ... (ln, i in ws_idxs or i in pass_idxs) ...
confidence: high
related_task: #1345
<!-- /workflow-fix-candidate -->
