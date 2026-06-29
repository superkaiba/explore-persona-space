---
title: 'infra: harden shared HF uploader against transient 504 commit timeouts'
kind: infra
tags:
- hf-upload-robustness
created_at: '2026-06-29T18:48:57Z'
has_clean_result: false
origin_prompt: 'Also file and run an infra task to figure out why this happened and
  fix it (HF 504 crashed #658 upload; shared orchestrate/hub.py uploader has the same
  no-retry gap)'
---
## Overview / Motivation

A persona-vectors follow-up on #658 crashed and burned an idle GCP A100 because
its `upload_pv_store` did ONE `HfApi.upload_folder` commit of a many-file store
with NO transient retry — only storage-quota-403 was caught. HF's
`/api/datasets/.../commit/main` endpoint returned a **504 Gateway Time-out** under
load; the un-retried exception crashed the whole workload, which then relaunched
on a fresh GPU instance (idle, since upload is network work) and risked looping.
A one-off retry wrapper was added to the issue658 script (commit `62bb233cd0` on
branch `issue-658`), but the SHARED uploader in `orchestrate/hub.py` has the
identical gap, so this 504 can crash ANY experiment's upload.

## Goal

Root-cause the transient-HF-504-crashes-the-workload failure mode and fix it AT
THE SHARED LAYER so every experiment's HF uploads survive transient commit
timeouts — not just the issue658 one-off. After this, a transient 504 on an HF
commit self-heals in-place (exp-backoff retry) instead of crashing the workload
and forcing a GPU relaunch.

## Root cause (to confirm + document in the clean-result / PR)

1. **No transient retry on the upload commit.** `orchestrate/hub.py`'s shared
   upload helpers (`_upload_folder_filtered`, `upload_dataset_directory`,
   `upload_model`, `upload_raw_completions_to_data_repo`, and any other
   `upload_folder` / `upload_file` call sites) wrap only specific errors
   (storage-quota-403) and let a transient 5xx/timeout propagate, crashing the
   caller. HF's commit endpoint 504s intermittently under load — a known
   transient.
2. **Single large commit.** A `upload_folder` of a many-file store (per-rollout ×
   per-layer `torch.save` tensors) is committed in one server-side operation;
   past some file-count/size the `/commit` endpoint gateway-times-out (504).
   Confirm whether the project's data repo size + file counts make this
   structural (commit too big) vs purely transient, and decide whether to also
   route large folders through `HfApi.upload_large_folder` (multi-commit,
   resumable) where the repo-subpath layout allows, or chunk into sub-folder
   commits.
3. **Crash → full GPU relaunch.** Because the upload runs inside the GPU workload
   and there's no retry, a transient 504 takes down the whole run; the router
   relaunches on a fresh GPU instance (idle during the upload phase). Confirm and
   note the cost amplification (idle-GPU churn).

## Proposed fix

- Add an exp-backoff transient-retry wrapper (5xx / 429 / timeout / connection
  errors) around the `upload_folder` / `upload_file` calls in the shared
  `orchestrate/hub.py` helpers, generalizing the issue658 `_upload_folder_with_retry`
  pattern (commit `62bb233cd0`). Storage-quota-403 must STILL re-raise immediately
  so the existing overflow-repo fallback fires (do not swallow it).
- Evaluate `HfApi.upload_large_folder` (available in `huggingface_hub` 0.36.2) for
  large many-file uploads, or chunked-by-subdir `upload_folder` commits, where the
  current code targets a repo sub-path (`upload_large_folder` has no `path_in_repo`
  — handle that constraint).
- Keep the #664 invariant: bulk uploads use folder-level commits (NOT a per-file
  loop, which 504-storms on the recursive tree-listing). The retry/large-folder
  change must not reintroduce per-file uploads.
- Add a unit test (mock the HF API to raise a transient 504 then succeed; assert
  the wrapper retries and the quota-403 path still routes to overflow) under
  `tests/`.

## Scope / surfaces

- Primary: `src/explore_persona_space/orchestrate/hub.py` (the shared upload
  helpers + their call sites).
- A new/extended test under `tests/` for the retry + quota-403-passthrough
  behavior.
- This is LIBRARY infra (NOT workflow surface — `hub.py` is not in the
  workflow-fix scope list), so it runs as an ordinary `kind: infra` task through
  the full `/issue` code-change pipeline.

## Constraints / invariants

- Do not weaken the existing storage-quota-403 → overflow-repo fallback
  (`.claude/rules/upload-policy.md`).
- Preserve the #664 one-folder-commit (no per-file loop) invariant.
- `uv run ruff check` clean on touched files; relevant `tests/` pass.
- Reference the issue658 one-off (`62bb233cd0`) as the pattern; once the shared
  helper is hardened, the one-off can stay (harmless) or be noted as superseded.

## Reference

- issue658 one-off fix: commit `62bb233cd0` on branch `issue-658`
  (`scripts/issue658_extract_rb_personavectors.py::_upload_folder_with_retry`).
- `.claude/rules/upload-policy.md` (256-commits/hr cap, #664 per-file storm,
  quota-403 recovery) — the existing upload-robustness rules this extends.
