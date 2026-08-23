---
title: 'workflow-fix: step9c compare SCAN-NEW-VIOLATION false block — tb=short traceback
  location line enters the branch-side violation set (verbosity asymmetry vs pristine)'
kind: infra
tags:
- wf-fix
- step9c-gate
created_at: '2026-08-23T11:56:29Z'
has_clean_result: false
origin_prompt: 'Found by the #2305 orchestrator: compare stderr showed SCAN-NEW-VIOLATION
  naming the test''s own file as the branch-added violation path; branch tree byte-identical
  to main on both scan-relevant paths.'
workflow: v1
---
# step9c_baseline SCAN-NEW-VIOLATION false block: gate/pristine verbosity asymmetry lets the `--tb=short` traceback location line enter the violation-path set

## Goal

Close a false-BLOCK channel in the Step 9c compare's #2316/#2319 violation-set diff: the gate pytest runs with `-v --tb=short` (the selector's printed command) while the pristine oracle runs `PYTEST_BASE_FLAGS` (`-q --tb=no`), so for any `VIOLATION_SET_SCAN_NODES` member red on BOTH sides, the branch-side junit failure text contains pytest's traceback LOCATION line — `tests/test_shared_vm_thread_caps.py:1000: AssertionError` — which LEADS with a tracked path and passes the row-anchored `_VIOLATION_ROW_RE` (`scripts/step9c_baseline.py:381-384`; the `:` stops the capture at `.py`, so the extracted token is the test's own file). Pristine's `--tb=no` text has no such line. The set diff then reads `branch ⊃ pristine` and routes a SCAN-NEW-VIOLATION to `new` → COMPARE_RC=1 → gate FAIL, on a byte-identical scan population.

## Observed instance (#2305 gate, 2026-08-23)

`step9c_baseline: SCAN-NEW-VIOLATION: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints — branch adds violation path(s) absent on pristine main: ['tests/test_shared_vm_thread_caps.py']`

Mechanical refutation of the "branch adds" claim: `git -C <wt> diff origin/main -- tests/test_shared_vm_thread_caps.py scripts/issue823_shared_persona_paired.py` is EMPTY (both byte-identical to the main tip), so the branch cannot have added a violation. The genuine sole offender on both sides is `scripts/issue823_shared_persona_paired.py` (the pre-existing main red introduced by d526008c67, separately surfaced to #823).

## Blast radius

Every session's Step 9c gate, whenever ANY of the five `VIOLATION_SET_SCAN_NODES` members is red on main (as now) — the #1388 blast-radius class the #2319 anchoring fix was for. The extractor's own docstring sanitizers (assert-segment strip, `...` filter) do not cover the traceback location line; the #2319 curation criterion (d) audit only examined offender-row grammar, not the tb=short longrepr tail.

## Fix sketch (implementer's choice, smallest correct)

Either (a) add a sanitizer to `extract_violation_paths` dropping traceback-location rows (a row matching `^<path>:\d+: ` — path immediately followed by `:<lineno>:`), or (b) symmetrize the two sides by extracting from the junit `message` attribute only (no longrepr/traceback), or (c) strip the trailing traceback block under `--tb=short` before extraction. Preserve the #2319 anchored-row semantics and the fail-toward-strip direction; extend the pin tests (`tests/test_step9c_baseline.py`) with a tb=short-shaped fixture red on both sides that must NOT produce SCAN-NEW-VIOLATION.

## Provenance

Found by the #2305 orchestrator during its round-1 Step 9c compare (compare stderr + `scripts/step9c_baseline.py:334-442` source read). Not a #2305 deliverable — filed separately per the workflow-fix-on-bug protocol.
