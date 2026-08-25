---
title: 'workflow-fix: scripts/issue1739_r2v2_run.py has ungated HfApi upload call
  sites — test_no_ungated_upload_call_sites RED on origin/main'
kind: infra
tags:
- wf-fix
- ungated-upload-call-site
created_at: '2026-08-21T18:48:26Z'
has_clean_result: false
origin_prompt: 'Surfaced by the /issue 2260 unit-2 Step 9c pin sweep 2026-08-21: test_no_ungated_upload_call_sites
  is RED on pristine origin/main; scripts/issue1739_r2v2_run.py makes two direct HfApi().upload_file
  calls without the secret gate (landed via #1739 crash-fix 067bd0300c).'
workflow: v1
---
kind: infra

## Goal

Restore `tests/test_no_ungated_upload_call_sites.py::test_no_new_ungated_upload_call_sites` to green on `origin/main` by routing `scripts/issue1739_r2v2_run.py`'s two direct `HfApi().upload_file` call sites through the secret-scrub gate.

## Incident (reproduced evidence)

- The test FAILS on pristine `origin/main` (reproduced from the repo root at `origin/main` f067e7fba0, 2026-08-21):
  ```
  AssertionError: NEW direct HF upload call site(s) without the secret gate:
      scripts/issue1739_r2v2_run.py
  assert not ['scripts/issue1739_r2v2_run.py']
  tests/test_no_ungated_upload_call_sites.py:176
  ```
- Offending call sites: `scripts/issue1739_r2v2_run.py` ~L439 and ~L543 (direct `HfApi().upload_file`, no `secret_scrub.assert_upload_clean(...)` and not routed through `hub._upload` / `upload_dir_sharded`).
- Provenance of the regression: landed via the #1739 crash-fix commit `067bd0300c`. The file's blob is `b7fd9319914edb64b787efdc8a1508b98b6fc584` at both the merge-base and `origin/main`.

## Why this is filed separately

Surfaced by the #2260 unit-2 implementer while running the Step 9c-scoped pin sweep. It is NOT a #2260 regression: #2260's round diff never touches `scripts/issue1739_r2v2_run.py` nor `tests/test_no_ungated_upload_call_sites.py`, and the offending blob is identical at merge-base and `origin/main` (both verified by the #2260 orchestrator before filing). #2260's own Step 9c gate WILL select this test — it rides the selector's basename/literal arms — so the gate's baseline attribution is expected to absorb it, but the underlying main-side red should not be left standing behind that attribution.

## Fix direction (implementer to confirm the right shape)

Route both call sites through the gated helper (`hub._upload` / `upload_dir_sharded`), or call `secret_scrub.assert_upload_clean(paths, what=...)` immediately before each direct call. **Do NOT add the file to `GRANDFATHERED`** — the test's own docstring forbids it explicitly.

## Acceptance criteria

1. `uv run pytest tests/test_no_ungated_upload_call_sites.py -q` is green with no new `GRANDFATHERED` entry.
2. The upload behavior of `scripts/issue1739_r2v2_run.py` is unchanged apart from acquiring the secret gate (it is a landed experiment driver; do not alter its artifact paths or upload semantics).
3. No other ungated call site is introduced.

## Provenance

Surfaced by the `/issue 2260` session (2026-08-21) during the unit-2 Step 9c pin sweep; independently reproduced by the #2260 orchestrator at the repo root before filing. Fingerprint: (ungated-hf-upload-call-site, scripts/issue1739_r2v2_run.py).
