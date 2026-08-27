---
title: step9c_baseline scan-set extractor counts the failing test's own traceback
  header as a branch-added violation path (false SCAN-NEW-VIOLATION)
kind: infra
tags:
- wf-fix
created_at: '2026-08-27T18:22:37Z'
has_clean_result: false
origin_prompt: 'Surfaced by /issue 2365 Step 9c: compare rc=1 SCAN-NEW-VIOLATION named
  tests/test_shared_vm_thread_caps.py (the scan test''s own traceback header) as a
  branch-added violation path while the assertion''s violation set was identical on
  both sides.'
workflow: v1
---
# step9c_baseline.py scan-set violation-path extractor counts the failing test's own traceback header as a branch-added violation path (false NEW / SCAN-NEW-VIOLATION)

## Symptom (observed on #2365's Step 9c gate, 2026-08-27)

The #2316 scan-set arm (`scan_violation_diffs`) classified
`tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
as `new-violations` with
`new_violations: ["tests/test_shared_vm_thread_caps.py"]` — the scan test's OWN
file — flipping an otherwise-clean compare to rc=1 (NEW) on a branch that never
touched the file.

Ground truth: the branch-side junit failure text's assertion listed exactly ONE
violation, `scripts/issue2569_normalize_passb.py` — identical to the pristine
side's `pre_existing` set. The extra "violation path" came from the pytest
`--tb=short` traceback header line
`tests/test_shared_vm_thread_caps.py:1000: in test_no_new_torch_before_dotenv_vm_entrypoints`,
which the extractor's path-shaped-token grep captured on the branch side only
(the pristine oracle's own run/extraction did not include it — an
output-format asymmetry between the gate run and the pristine run).

## Fix direction (implementing session decides)

In `scripts/step9c_baseline.py`, the violation-path extraction for
`VIOLATION_SET_SCAN_NODES` members should parse only the assertion's violation
list (e.g. lines under the `assert not [...]` payload / the message attribute),
or at minimum exclude the failing node's own `file` attribute path and
`<path>:<line>: in <test name>` traceback-header-shaped lines. Apply the same
normalization to BOTH sides so gate-vs-pristine extraction is symmetric.
Add a regression test reproducing the #2365 shape: a junit failure whose text
carries a traceback header + a one-path assertion list must extract exactly
the one assertion path.

## Evidence

- Compare JSON: `/tmp/step9c-compare-issue-2365.json` (`scan_violation_diffs[0]`)
  — `pre_existing: [scripts/issue2569_normalize_passb.py]`,
  `new_violations: [tests/test_shared_vm_thread_caps.py]`, verdict
  `new-violations`; compare rc=1.
- Branch junit: `/tmp/step9c-junit-issue-2365.xml` — the failure element's
  assertion names only `scripts/issue2569_normalize_passb.py`.
- Git provenance: `git log origin/main..issue-2365 -- tests/test_shared_vm_thread_caps.py scripts/issue2569_normalize_passb.py`
  is EMPTY (branch never touched either file).
- #2365 events.jsonl carries the full disposition (provenance-override recorded
  in its `epm:test-verdict`).

## Acceptance

- The #2365 junit replayed through the fixed extractor yields equal violation
  sets on both sides (verdict `equal`, no `SCAN-NEW-VIOLATION`).
- New regression test passes; existing scan-set tests
  (`tests/test_workflow_lint*` / step9c baseline tests) stay green.
