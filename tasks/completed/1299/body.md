---
title: 'workflow-fix: step-9c selector import-map arm (stem-map misses import-relationship
  tests)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c4e9af27d15f
created_at: '2026-07-13T07:24:06Z'
has_clean_result: false
origin_prompt: 'Surfaced prose from #1286 Phase-2 statistics critic: select_step9c_tests.py''s
  stem-map misses tests that exercise a touched module under an unrelated filename
  — tests/test_issue810_uh_pack_validation.py not selected when issue810_common/fit_readout/bootstrap_deltaskill
  are the touched set.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1286 (emitting agent: #1286's Phase-2 statistics critic, surfaced
prose — "Selector-shape observation").

## Goal

Extend `scripts/select_step9c_tests.py` so the touched-scope selection also picks
test files whose module-top imports name a touched `scripts/` module
(import-graph mapping, or an explicit module→test registry), so touched-module
test coverage no longer depends on test-file NAMING.

## Workflow gap

- **Bug observed:** the touched-scope stem-map selector misses tests that exercise
  a touched module under an unrelated filename. Concrete instance (#1286):
  `tests/test_issue810_uh_pack_validation.py` imports `issue810_common`,
  `issue810_fit_readout`, and `issue810_bootstrap_deltaskill`, yet when exactly
  those scripts are the touched set, `select_tests_with_reasons()` does NOT select
  it — the stem globs (`tests/test_{stem}.py` exact + `tests/test_*{stem}*.py`
  broad) match nothing because "uh_pack_validation" contains no touched stem. A
  code change to those modules can therefore pass the step-9c gate without its
  most relevant test ever running (4 of the 5 touched scripts land only in the
  `untested_touched` WARN list). #1286's plan had to add a manual "append the
  target test explicitly to the gate invocation" binding as a workaround.
- **Why it is a workflow gap:** `select_step9c_tests.py` is the single source of
  the step-9c touched-scope selection (workflow surface); its stem-map maps by
  FILENAME similarity only, with no import-relationship arm, so the gate's
  coverage silently varies with test naming conventions.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "stem" scripts/select_step9c_tests.py` → stem-map
  documented at lines 24-34 (exact + broad `*{stem}*` glob arms, no import arm)
  (2026-07-13); the #1286 statistics critic re-ran `select_tests_with_reasons()`
  on the 6-file touched set the same day and confirmed the target test absent
  from the selection.

## Proposed change (candidate diff sketch — refine in planning)

```
+ In select_tests_with_reasons(): after the stem-map pass, add an
+ import-map pass — for each tests/test_*.py, parse module-top
+ `import X` / `from X import ...` statements (ast, cheap) and select
+ the test when X's resolved file is in the touched set; reason string
+ `import-map:<touched file>`.
+ Keep the stem map + GLOB_SCAN_TESTS arms unchanged; extend
+ recommended_timeout_s() sizing to count import-map selections.
+ Pin behavior in tests/test_select_step9c_tests.py (the #1286 shape:
+ touched issue810_common.py must select test_issue810_uh_pack_validation.py).
```

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'stem' scripts/select_step9c_tests.py tests/test_select_step9c_tests.py`)
  and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- Selection may only GROW (safe-by-direction: over-selection costs minutes,
  under-selection ships regressions); the import-map arm must never remove a
  stem-map/GLOB_SCAN selection.
- Keep `select_step9c_tests.py --map-files` (the #1147 lint-gate leg contract)
  byte-compatible.

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: c4e9af27d15f

Surfaced prose (verbatim, from #1286's Phase-2 statistics-critic report):
"Selector-shape observation (for the orchestrator, no candidate block —
AUTO_REVIEW_DISABLED): `select_step9c_tests.py`'s stem-map misses tests that
exercise a touched module under an unrelated filename — exactly the
fleet-breaking test here."
