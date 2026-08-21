---
title: 'workflow-fix: slow-tests roster misses the live-bundle no-flags lint test'
kind: infra
tags:
- wf-fix
- wf-fix-fp:slow-roster-upload-or-true
created_at: '2026-08-21T11:19:16Z'
has_clean_result: false
origin_prompt: 'implementer at #2253 impl round 1, formal workflow-fix-candidate:
  tests/test_workflow_lint_upload_or_true.py::test_no_flags_default_run_bundles_check
  runs a full live-tree no-flags workflow_lint.main([]) in-process (>533s measured)
  but is absent from select_step9c_tests.py''s slow-tests roster, so an implementer''s
  gate-matched local run deterministically timeout-kills on it under the 600s Bash
  cap (three killed attempts on #2253 r1).'
workflow: v1
---
# workflow-fix: slow-tests roster misses the live-bundle no-flags lint test

## Goal

Stop `tests/test_workflow_lint_upload_or_true.py::test_no_flags_default_run_bundles_check` from deterministically timeout-killing an implementer's local Step-9c-matched pytest run, by either registering the file in `scripts/select_step9c_tests.py`'s slow-tests roster with a measured surcharge, or refactoring the test to the sub-second fixture-tree form its own sibling already uses.

## The gap

That test executes a FULL live-tree no-flags `workflow_lint.main([])` **in-process**. Measured: **>533 s for that single test**; the whole live bundle runs ~25-35 min. The file is ABSENT from `select_step9c_tests.py`'s slow-tests roster, so the selector neither defers it nor sizes a surcharge for it — and an implementer running the gate-matched selection locally under the 600 s Bash cap gets a deterministic `rc=124` timeout kill with no indication of which test stalled.

The roster exists precisely to prevent this class. The omission is the bug.

## Driving incident — #2253 round 1 (2026-08-21)

The #2253 implementer hit **three consecutive timeout-killed chunk attempts** before isolating the offender via `--collect-only` index-splitting. Evidence recorded in that round's `epm:results v1`:

- `/tmp/i2253-upload-b.out` — `rc=124` on a 10-test slice, stall at the 8th id.
- `tests/test_workflow_lint_upload_or_true.py:465-473` — no `_REPO_ROOT` monkeypatch, so the test builds no fixture tree and runs against the real repo.

Note the failure mode is *silent-by-construction*: a timeout kill names no test, so each attempt costs a full 600 s before yielding any diagnostic. That is what turned one slow test into three wasted cycles.

## Two candidate fixes (the implementing session picks; both are in-scope)

1. **Roster registration** — add `tests/test_workflow_lint_upload_or_true.py` to the slow-tests roster with a MEASURED surcharge (the >533 s figure above is a floor, not a basis; measure at the production entrypoint). Cheapest, preserves the test's current live-tree coverage semantics.
2. **Refactor to the fixture-tree form** — monkeypatch `wl._REPO_ROOT` onto a tmp fixture tree exactly as the sibling exemplar `tests/test_workflow_lint_scripts_import_guard.py::test_check_bundled_in_no_flags` already does. Sub-second, and it retains the same mutation-visibility property (deleting the `or no_flags` branch still fails the test). Strictly better if the live-tree read is not load-bearing — verify that before choosing it.

Prefer (2) if the live-tree execution is incidental; prefer (1) if it is deliberate coverage. Do not do both.

## Acceptance

- An implementer's gate-matched local run over a diff selecting this file completes without a timeout kill, OR the selector defers the file with a measured surcharge and says so.
- If fix (2): the test still FAILS when the `or no_flags` bundling branch is deleted (mutation check), proving coverage was preserved rather than dropped.
- If fix (1): `select_step9c_tests.py --json` reports the file in `slow_tests_selected` with a non-default `recommended_timeout_s`, and a pin test asserts the roster membership so a future roster edit cannot silently drop it.
- No other selector behavior changes; the existing roster pin tests stay green.

## Related

- `scripts/select_step9c_tests.py` (the slow-tests roster — the target surface)
- `tests/test_workflow_lint_scripts_import_guard.py::test_check_bundled_in_no_flags` (the sub-second exemplar to copy)
- `#2253` (driving incident; its `epm:results v1` carries the full evidence and the NOT-RUN single-test record)

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- candidate-fingerprint: slow-roster-missing-upload-or-true-live-bundle-test

Surfaced as a formal `<!-- workflow-fix-candidate v1 -->` block by the `implementer` at #2253 implementation round 1, and routed by the #2253 orchestrator per the workflow-fix-on-bug protocol. The orchestrator verified the two claims that matter: the file is genuinely absent from the roster, and neither this test nor the file is present in #2253's own 134-test Step 9c selection (`base_identical_excluded`), so #2253's own gate is unaffected — this is filed for the fleet, not to unblock #2253.
