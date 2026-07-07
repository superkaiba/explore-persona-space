---
title: 'workflow-fix: verify_task_body warn on body-linked figures untracked at branch
  HEAD'
kind: infra
tags:
- wf-fix
- wf-fix-fp:aee0b9bd1097
created_at: '2026-07-04T07:09:30Z'
has_clean_result: false
origin_prompt: 'prose follow-up from interpretation-critic on #841 (interp-critique
  v5)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the `interpretation-critic` on task #841 (interp-critique v5, 2026-07-04).

## Goal

Add a warn-level check to `scripts/verify_task_body.py` that every body-linked `figures/issue_<N>/...` path also resolves in `git ls-tree` at the issue-branch HEAD (not only at the pinned SHA), so silent figure-tracking loss cannot hide behind immutable SHA-pinned raw.githubusercontent URLs.

## Workflow gap

- **Bug observed:** three body-linked parent figure stems (`hero_matched_information`, `exploratory_atlas_meancentered`, `exploratory_transport_fidelity_cosine` under `figures/issue_841/`) were tracked at the pinned SHA `4824a567aa` but UNTRACKED at branch HEAD — local copies byte-identical, so every SHA-pinned body URL still rendered and nothing surfaced the loss.
- **Why it is a workflow gap:** `verify_task_body.py` checks that figure URLs resolve (HTTP) and are SHA-pinned, but a pinned URL is immutable — it keeps resolving forever even after the file falls out of tracking on the live branch, so a later merge/rebase can silently drop the canonical figure files from `main` with zero verifier signal.
- **Confidence (emitter):** high (the incident is concrete and reproducible; the check is mechanizable).

## Proposed change (candidate diff sketch — refine in planning)

+ In verify_task_body.py, for each body-linked raw.githubusercontent figure URL matching figures/issue_<N>/...:
+   path = extract repo-relative path from URL
+   present_at_head = `git ls-tree -r --name-only <issue-branch-or-HEAD> -- <path>` non-empty
+   (resolve the branch as issue-<N> when it exists, else HEAD of the checkout the verifier runs in)
+   absent -> WARN (not FAIL): "body-linked figure <path> tracked at pinned SHA but missing at branch HEAD — re-commit it"
WARN-level only (the pinned URL still renders; this is drift hygiene, not a broken body).

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Also touch its tests (`tests/test_verify_task_body.py`) with a fixture for the tracked-at-SHA-untracked-at-HEAD case.

## Constraints / invariants

- Workflow-surface only. WARN, never FAIL (grandfathered bodies with fully-merged branches must not regress).
- `scripts/workflow_lint.py --check-asks` passes; ruff passes; existing verify_task_body tests stay green.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: aee0b9bd1097

Surfaced prose (verbatim, from the interpretation-critic's report on #841): "Suggest `scripts/verify_task_body.py` gain a warn-level check that every body-linked `figures/issue_<N>/...` path also resolves in `git ls-tree` at the issue-branch HEAD, so silent figure-tracking loss can't hide behind immutable pinned URLs; also re-commit the three files on `issue-841`." (The re-commit half was executed directly by the orchestrator: commit c7eba35ada on issue-841.)
