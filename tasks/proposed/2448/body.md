---
title: 'Fleet-wide Step 9c red: two ungated HF upload call sites in scripts/issue1739_r2v2_run.py'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ungated-upload-issue1739-r2v2-run
created_at: '2026-08-21T11:20:10Z'
has_clean_result: false
origin_prompt: 'implementer at #2253 impl round 1: tests/test_no_ungated_upload_call_sites.py
  FAILS on pristine origin/main because scripts/issue1739_r2v2_run.py:439,543 call
  HfApi().upload_file without the secret gate; byte-identical on origin/main with
  zero commits since #2253''s base, so any round selecting that invariant test bounces
  its Step 9c gate on a red it cannot fix. Orchestrator re-verified via origin/main
  blob read + live pytest (1 failed).'
workflow: v1
---
# Fleet-wide Step 9c red: two ungated HF upload call sites in scripts/issue1739_r2v2_run.py

## Goal

Restore `tests/test_no_ungated_upload_call_sites.py` to green on `main` by routing the two direct `HfApi().upload_file` calls in `scripts/issue1739_r2v2_run.py` through the gated path (or asserting the secret gate before them), so that any round whose Step 9c selection includes that invariant test stops bouncing on a failure it did not cause.

## The problem — standing red on main, not a diff regression

`scripts/issue1739_r2v2_run.py` calls `HfApi().upload_file(...)` directly at **line 439** (`_stage_crumb_path` upload) and **line 543** (`_gate1_crumb_path` upload), both wrapped in `hub.retry_transient(...)` but neither routed through the gated `hub._upload` / `upload_dir_sharded` path and neither preceded by `secret_scrub.assert_upload_clean(...)`.

Verified by the #2253 orchestrator (2026-08-21):

- `git show origin/main:scripts/issue1739_r2v2_run.py` — both call sites present, byte-identical, on **pristine `origin/main`**.
- `git log 2087dda0c1..origin/main -- scripts/issue1739_r2v2_run.py` — **zero** commits since #2253's worktree base, so this is not a recent regression and is not attributable to any in-flight round.
- Live run: `uv run pytest tests/test_no_ungated_upload_call_sites.py -x -q` → **1 failed**, `AssertionError: NEW direct HF upload call site(s) without the secret gate: scripts/issue1739_r2v2_run.py`.

## Why this is worth its own task rather than a note

The failing test is a WORKFLOW INVARIANT test, so it is stem-mapped into the Step 9c selection of any round touching the upload surface. Every such round bounces its test-verdict gate on a red it cannot fix within its own scope — the #1388 shape, except the red is already landed rather than newly introduced. The cost is paid repeatedly by unrelated tasks until someone gates these two calls.

The test's own docstring is explicit that the `GRANDFATHERED` allowlist is not the remedy ("Never add to GRANDFATHERED — see this file's docstring"), so the fix is to gate the calls.

## Proposed change

Route both call sites through the gated upload path — `hub._upload` / `upload_dir_sharded` — or, if the direct call is deliberate for these small JSON crumb files, call `secret_scrub.assert_upload_clean(paths, what=...)` immediately before each. Follow whichever pattern the file's other upload sites (if any) already use, and whatever `hub` exposes for a single small file, rather than inventing a third shape.

Both payloads are small JSON stage/gate crumb files written to `hub.DEFAULT_DATASET_REPO` as `repo_type="dataset"`, so neither is an LFS-path or a bulk-upload case.

## Acceptance

- `uv run pytest tests/test_no_ungated_upload_call_sites.py` PASSes with no `GRANDFATHERED` addition.
- The two crumb uploads still function (the gate is added, the upload is not removed) — confirm via the file's own smoke path or a dry-run, not by deleting the calls.
- `scripts/workflow_lint.py` no-flags run stays green; `ruff` clean on the touched file.

## Related

- `scripts/issue1739_r2v2_run.py:439,543` (the two call sites)
- `tests/test_no_ungated_upload_call_sites.py:176` (the failing assertion)
- `#1739` (the owning experiment task — the script is its driver)
- `#2253` (surfacing round; its implementer flagged this while running its own gate scope, with the origin/main byte-identity proof)

## Provenance

Surfaced by the `implementer` at #2253 implementation round 1 as one of two orchestrator-routing items, and independently re-verified by the #2253 orchestrator (origin/main blob read, empty commit log since base, live pytest failure) before filing. Filed rather than left as a chat note because the red is standing on `main` and taxes every future round that selects the test — #2253's own 134-test selection does NOT include it, so this is filed for the fleet, not to unblock #2253.
