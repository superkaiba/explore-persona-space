---
title: 'workflow-fix: two ungated HfApi().upload_file call sites in issue1739_r2v2_run.py
  bypass the secret-scrub gate (test RED on main)'
kind: infra
tags: []
created_at: '2026-08-22T07:37:00Z'
has_clean_result: false
origin_prompt: 'Discovered during /issue 2271 Step 4b gate-scope enumeration (2026-08-22):
  tests/test_no_ungated_upload_call_sites.py is RED on pristine main with offender
  scripts/issue1739_r2v2_run.py (lines 439/543 call HfApi().upload_file with no assert_upload_clean
  gate, landed via #1739 crash-fix rounds). Reddens the Step 9c gate for any unrelated
  task whose selector pulls in this invariant test. Fix is wiring the gate, never
  a GRANDFATHERED entry.'
workflow: v1
---
# Two ungated `HfApi().upload_file` call sites in `scripts/issue1739_r2v2_run.py` bypass the secret-scrub gate (`test_no_ungated_upload_call_sites` RED on main)

## Goal

Route the two direct `HfApi().upload_file` calls in
`scripts/issue1739_r2v2_run.py` through the secret-scrub gate — either the
gated `hub._upload` / `upload_dir_sharded` helpers, or an explicit
`secret_scrub.assert_upload_clean(paths, what=...)` immediately before each
direct call — so no Hub-bound payload leaves the fleet unscanned and
`tests/test_no_ungated_upload_call_sites.py` returns to green on `main`.

## Current state — RED on pristine main

    $ uv run pytest tests/test_no_ungated_upload_call_sites.py -x -q
    E   AssertionError: NEW direct HF upload call site(s) without the secret gate:
    E       scripts/issue1739_r2v2_run.py
    FAILED tests/test_no_ungated_upload_call_sites.py::test_no_new_ungated_upload_call_sites

The two offending call sites:

    scripts/issue1739_r2v2_run.py:439        lambda: HfApi().upload_file(
    scripts/issue1739_r2v2_run.py:543        lambda: HfApi().upload_file(

Neither is preceded by an `assert_upload_clean` call (grep for it in that file
returns nothing).

## Provenance — a crash-fix round, not a deliberate bypass

The file landed through #1739's claim-4 crash-fix rounds (most recently
`067bd0300c`, "row-index pushdown kills the 52 GiB split-copy stack"; earlier
`95a838963e`, `af860da3ad`). Crash-fix rounds ship under time pressure against
a live billing pod, which is exactly the channel by which an ungated upload
helper slips past — the round's own review scope was the OOM fix, not the
upload seam. No sign of an intentional bypass; the gate simply was not wired.

## Why it needs its own task

1. **It is a live secret-exposure seam, not a lint nit.** The gate exists so no
   Hub-bound text is uploaded unscanned (`9563d9f8d7` "Secret upload gate +
   scrub tool: no Hub-bound text leaves unscanned"). Two call sites currently
   upload without it.
2. **It reddens every task's Step 9c gate that selects this test.** Discovered
   during `/issue 2271`'s Step 9c gate-scope enumeration: #2271's diff touches
   only `orchestrate/hub.py` + `tests/test_hub.py` and does NOT touch
   `scripts/issue1739_r2v2_run.py`, yet the selector pulls this invariant test
   in, so an unrelated task inherits a red gate it cannot fix in scope. Any
   other task whose selection includes this test inherits the same.
3. **The fix must NOT be a grandfather entry.** The test's own docstring says
   `Never add to GRANDFATHERED`. The remedy is wiring the gate, not widening
   the allowlist.

## Acceptance

1. Both call sites (`:439`, `:543`) route through the gate — via
   `hub._upload` / `upload_dir_sharded`, or an explicit
   `secret_scrub.assert_upload_clean(paths, what=...)` immediately before the
   direct call. `GRANDFATHERED` is NOT touched.
2. `uv run pytest tests/test_no_ungated_upload_call_sites.py` PASSES on `main`.
3. `uv run python scripts/workflow_lint.py` (no flags) passes; `ruff` clean on
   the changed file.
4. The change is upload-path wiring only — no change to what the script
   computes, uploads, or where it uploads to. `scripts/issue1739_r2v2_run.py`
   is a per-issue reproducibility driver, so keep the diff minimal and state
   in the report that the uploaded payload and destination prefix are
   unchanged.

## Notes for the implementer

- Read the gate's own contract first (`secret_scrub.assert_upload_clean` and
  the `hub._upload` wrapper) and prefer the wrapper where the call shape
  allows — a wrapper swap inherits future gate improvements, an inline assert
  does not.
- Both sites are wrapped in `lambda:` (a retry envelope), so place the gate
  call OUTSIDE the lambda — inside, it would re-scan on every retry, and a
  scan failure would be misread as a transport failure by the retry logic.
- #1739 is a large multi-round task; do not attempt to re-run or re-validate
  its experiment. This is a two-call-site wiring fix.

## Provenance

workflow_fix_target: scripts/issue1739_r2v2_run.py

Discovered 2026-08-22 during `/issue 2271` Step 4b, when the implementer's
mandatory gate-scope enumeration ran the selected invariant tests locally and
found this one red on pristine `main`. Orchestrator-verified independently:
the test fails at the repo root with the offender named, and #2271's diff
touches neither the offending script nor the test.
