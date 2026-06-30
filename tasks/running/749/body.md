---
title: 'workflow-fix: document HF recursive tree listing 504 un-retry in gotchas.md'
kind: infra
tags:
- wf-fix
- wf-fix-fp:hf-recursive-tree-listing-pagination-5xx-untretried
created_at: '2026-06-30T05:36:29Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from #658 r3: huggingface_hub.paginate retries
  only 429, not 5xx, on list_repo_files cursor pages'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised by the experiment-implementer on task #658 (round-3 fix after a recurring HF Hub 504 on the post-upload verify-listing endpoint).

## Goal

Document in `.claude/rules/gotchas.md` that HF Hub's `list_repo_files` / `list_repo_tree(recursive=True)` paginate `/api/.../tree/main` and the underlying `huggingface_hub.utils._pagination.paginate` retries ONLY on 429 (NOT 5xx) — any 504 on a follow-up cursor page propagates straight through, even on otherwise-retry-wrapped upload pipelines.

## Workflow gap

- **Bug observed:** task #658 r3 — the `upload_folder` retry caught 504s on `commit/main`, but a SEPARATE post-upload `list_repo_files(recursive=True)` verify call hit `/tree/main` 504s on cursor pages 2+; `huggingface_hub.paginate` only retries 429, so the run crashed rc=1 AFTER successfully committing 12081 files.
- **Why workflow gap:** `.claude/rules/gotchas.md` + `.claude/rules/upload-policy.md` document the 256-commits/hr throttle and storage-quota-403, but say nothing about pagination's 429-only retry — every HF-upload-heavy script re-introduces this trap.
- **Confidence:** medium (the implementer raised it with confidence medium — the pattern is real but the exact set of HF endpoints affected may broaden).

## Proposed change

Add a single bullet to `.claude/rules/gotchas.md`'s HF section:

```
- **HF recursive tree listing 504s are un-retried.** `list_repo_files` /
  `list_repo_tree(recursive=True)` paginate `/api/.../tree/main?...&cursor=...`;
  `huggingface_hub.utils._pagination.paginate` retries ONLY 429 on follow-up
  cursor pages (`http_backoff(..., retry_on_status_codes=429)`), so a 504 on
  any cursor page propagates straight through, even when the upload itself is
  retry-wrapped. Wrap any post-upload verify listing on a large repo in the
  SAME transient-5xx retry as the upload (#658).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Verify no existing reference: `grep -rln "tree/main\|list_repo_files\|list_repo_tree" .claude/ CLAUDE.md scripts/`.

## Constraints / invariants

- Workflow-surface only — documentation addition.
- `scripts/workflow_lint.py` passes; ruff (no .py touched) passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (and carries a `workflow_fix_target:` Provenance line) — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: hf-recursive-tree-listing-pagination-5xx-untretried

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: task #658 r3 — post-upload `list_repo_files(recursive=True)` verify call hit a 504 on `/tree/main` cursor page 2+; `huggingface_hub.utils._pagination.paginate` retries only 429 on follow-up pages, so the 504 propagated straight through despite the upload itself being retry-wrapped. Run crashed rc=1 after successfully committing 12081 files.
why_workflow_gap: gotchas.md + upload-policy.md document the 256-commits/hr throttle and the storage-quota-403, but say nothing about pagination's 429-only retry. Every HF-upload-heavy script re-introduces this trap (the 5h compute on #658 was nearly stranded on it).
proposed_change: add a single bullet to .claude/rules/gotchas.md naming the 429-only retry behavior of paginate() on tree/main and the wrap-the-verify-call default for any post-upload listing on a large repo.
diff_sketch: |
  + - **HF recursive tree listing 504s are un-retried.** `list_repo_files` /
  +   `list_repo_tree(recursive=True)` paginate `/api/.../tree/main?...&cursor=...`;
  +   `huggingface_hub.utils._pagination.paginate` retries ONLY 429 on follow-up
  +   cursor pages (`http_backoff(..., retry_on_status_codes=429)`), so a 504 on
  +   any cursor page propagates straight through. Wrap any post-upload verify
  +   listing on a large repo in the SAME transient-5xx retry as the upload
  +   (#658).
confidence: medium
related_task: #658
<!-- /workflow-fix-candidate -->
