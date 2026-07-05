---
title: 'workflow-fix: verify_task_body WARN on HF file-count claims vs scoped listing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:16d3f1799c7b
created_at: '2026-07-04T11:45:03Z'
has_clean_result: false
origin_prompt: 'Formal workflow-fix-candidate from clean-result-critic on #931: file-count
  claims (528/10/3/198) vs Hub listing (515/9/2/197) — folder entries counted as files;
  add WARN-level mechanical check'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a formal candidate block raised on task #931 (emitting agent: clean-result-critic).

## Goal

Add a WARN-level verify_task_body check comparing numeric file-count claims near HF /tree links against a scoped list_repo_tree RepoFile count.

## Workflow gap

- **Bug observed:** #931's body claimed "528 files" / "10 shards" / "3 files" / "198 files" for HF artifacts where the scoped listing at the pinned revision holds 515/9/2/197 — `list_repo_tree` folder entries were counted as files, and no mechanical check compares count claims against the Hub.
- **Why it is a workflow gap:** The verifier already resolves HF URL pins (check "HF URL pins resolve at the cited revision") but never validates adjacent numeric file-count claims, so a systematic entry-vs-file miscount ships unless an LM critic re-lists the repo.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ def check_hf_file_count_claims(body, ...):
+     for (count, prefix, sha) in _extract_count_claims_near_hf_tree_links(body):
+         n_files = sum(1 for e in list_repo_tree(repo, path_in_repo=prefix,
+                       repo_type="dataset", revision=sha, recursive=True)
+                       if isinstance(e, RepoFile))
+         if count != n_files:
+             warn(f"body claims {count} files at {prefix}, Hub has {n_files} "
+                  f"(RepoFolder entries are not files)")

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'list_repo_tree' scripts/verify_task_body.py .claude/`) and update every hit; list them in the plan. Network-dependent check must be fail-soft/skippable offline (WARN-level, never a blocking FAIL on a network error).

## Constraints / invariants

- Workflow-surface only. `tests/test_verify_task_body.py` extended + green; ruff clean.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 16d3f1799c7b

Verbatim candidate block from clean-result-critic on #931: see epm:clean-result-critique v1 marker (workflow-fix-candidate v1 targeting scripts/verify_task_body.py, WARN-level HF file-count claim check, RepoFolder-entries-are-not-files).
