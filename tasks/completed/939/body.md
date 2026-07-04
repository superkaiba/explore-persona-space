---
title: 'workflow-fix: scoped listing in verify_uploads.py (1M-file repo hang)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:df81c7657307
created_at: '2026-07-03T15:46:28Z'
has_clean_result: false
origin_prompt: 'upload-verifier #920 finding: verify_uploads.py line ~152 bare list_repo_files
  hangs >600s on the ~1M-file data repo (#833 gotcha); fix = scoped list_repo_tree(path_in_repo=...)/file_exists
  per path'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a finding surfaced by the `upload-verifier` on task #920 (formal candidate suppressed agent-side under AUTO_REVIEW_DISABLED; routed by the orchestrator).

## Goal

Replace `scripts/verify_uploads.py`'s bare full-repo `api.list_repo_files(...)` (dataset-path + claimed-urls verification, ~line 152) with scoped `list_repo_tree(path_in_repo=<prefix>)` / `HfApi().file_exists(...)` per checked path.

## Workflow gap

- **Bug observed:** `verify_uploads.py` hung >600 s (killed, exit 143) on task #920 — its verification enumerates the ~1M-file `superkaiba1/explore-persona-space-data` repo with a bare `list_repo_files`, which times out (the #833 gotcha, now also codified in `.claude/rules/artifact-reuse.md`).
- **Why it is a workflow gap:** the upload-verifier's own mechanical helper cannot complete on the project's primary data repo; every experiment's Step 8 hard gate degrades to agent-side manual checks.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
- remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
+ # scoped per checked prefix — the ~1M-file data repo times out on a bare listing (#833)
+ remote = set()
+ for prefix in checked_prefixes:
+     remote |= {e.path for e in api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=prefix, recursive=True)}
+ # single-path checks: api.file_exists(repo_id, path, repo_type="dataset")
```

## Scope / surfaces

- Primary target: `scripts/verify_uploads.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'list_repo_files' scripts/ .claude/`) and audit every hit against the scoped-listing rule; list them in the plan.

## Constraints / invariants

- Workflow-surface only. Ruff passes; existing verify_uploads tests pass; behavior identical on small repos.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_uploads.py
- fingerprint: df81c7657307

Surfaced prose (upload-verifier #920, verbatim): "`scripts/verify_uploads.py` hung >600 s (killed, exit 143): its dataset-path + claimed-urls verification at `scripts/verify_uploads.py:152` uses a bare full-repo `api.list_repo_files(...)` against the ~1M-file data repo — the #833 gotcha. ... fix: scoped `list_repo_tree(path_in_repo=...)` / `file_exists` per path."
