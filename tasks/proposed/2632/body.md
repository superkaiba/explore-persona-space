---
title: step9c scan-violation set-diff extracts --tb=short traceback header as branch-added
  violation path (false gate BLOCK)
kind: infra
tags:
- workflow-fix
created_at: '2026-08-27T13:54:54Z'
has_clean_result: false
parent_id: 2624
origin_prompt: 'Found during #2624 step-9c gate: compare rc=1 with a single SCAN-NEW-VIOLATION
  naming the scan test''s own file, manufactured by the gate/pristine tb-format asymmetry
  (gate -v --tb=short vs pristine -q --tb=no).'
workflow: v1
---
---
kind: infra
---

# Step-9c scan-violation set-diff extracts the `--tb=short` traceback header as a branch-added violation path (false NEW / false gate BLOCK)

## Goal

Make `scripts/step9c_baseline.py::extract_violation_paths` (the #2316/#2319 scan-violation
set-diff) immune to the gate-vs-pristine traceback-format asymmetry, so a
`VIOLATION_SET_SCAN_NODES` member red on BOTH sides with an IDENTICAL offender set can never be
classified `SCAN-NEW-VIOLATION` off a traceback header line.

## Observed (issue #2624 step-9c gate, 2026-08-27)

The #2624 gate (206 files) failed compare with rc=1: exactly one `new` node,
`tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`, with
`scan_violation_diffs` claiming the branch adds violation path `tests/test_shared_vm_thread_caps.py`
— the scan test's OWN file. The branch touches neither that file nor the real offender
(`scripts/issue2569_normalize_passb.py`, pre-existing on main since 432da6f0c2 / #2569).

Mechanism, reproduced against the actual gate junit:

- `extract_violation_paths(failure message attr)` = `{scripts/issue2569_normalize_passb.py}` —
  correct, matches the pristine side.
- `extract_violation_paths(failure element text)` ADDS `tests/test_shared_vm_thread_caps.py`,
  extracted from the `--tb=short` traceback HEADER line
  `tests/test_shared_vm_thread_caps.py:1000: in test_no_new_torch_before_dotenv_vm_entrypoints`,
  which leads with a path token and matches `_VIOLATION_ROW_RE` (line-start anchor).
- The pristine oracle runs `PYTEST_BASE_FLAGS = ("-q", "--tb=no", ...)` (step9c_baseline.py:473),
  so the pristine failure text has NO traceback header. Gate side (`-v --tb=short`, per the
  canonical launcher) always has one. The set-diff therefore manufactures a NEW violation naming
  a non-offender on EVERY gate run that hits a red scan node — exactly the "false BLOCK on the
  gate that adjudicates every session's Step 9c" blast-radius class the #2316 docstring itself
  names (#1388).

#2624 resolved its gate via a manual provenance override recorded in its `epm:test-verdict v1`
(evidence: untouched files + reproduced extraction); every future branch hitting a red scan node
pays the same false FAIL until this lands.

## Requested change

In `extract_violation_paths` (or its caller), drop traceback-header rows before matching — e.g.
skip lines matching `^\S+\.py:\d+: in \S+` (the pytest short-tb header shape), or extract from
the junit `message` attribute ONLY for both sides (the message attr carries the full assertion
message including offender rows on both tb formats, and is what the pristine side effectively
reduces to). Add a regression fixture reproducing the #2624 shape: same offender set on both
sides, gate side with a short-tb header, assert verdict is NOT `new-violations`.

Keep the #2319 anchored-row semantics and both retained sanitizers unchanged for genuine
offender rows.

## Provenance

Found by the #2624 orchestrator session during its step-9c gate (compare rc=1, 2026-08-27
~13:30Z); evidence + reproduction recorded in #2624's `epm:test-verdict v1` marker and the
`[divergence-probe]`/gate breadcrumbs on #2624 events.jsonl. Gate junit preserved at
/tmp/step9c-junit-issue-2624.xml (VM-local, ephemeral).
