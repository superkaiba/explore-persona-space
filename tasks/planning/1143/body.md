---
title: 'workflow-fix: verify_task_body file-count claim-matcher misses footer shape'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d8e34254b667
created_at: '2026-07-08T09:31:23Z'
has_clean_result: false
origin_prompt: 'clean-result-critic r1 prose follow-up on #1112: the HF file-count
  check misses ''(N files: ...)'' after a backticked sub-path in a footer row with
  a pinned /tree/<sha> link; widen the matcher + count RepoFile entries only'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a clean-result-critic
prose follow-up on task #1112 (2026-07-08).

## Goal

Widen `verify_task_body.py`'s "HF file-count claims match the Hub tree"
claim-matcher to reach a parenthetical count following a backticked
sub-path within the same footer row as a pinned `/tree/<sha>` link, and
compare it against a recursive scoped `list_repo_tree` RepoFile count at
the cited revision.

## Workflow gap

- **Bug observed:** #1112's `**Repro:**` footer carried "(7,372 files:
  capture, tier-2, marker, read-out-extraction rollout text)" against a
  Hub tree that actually holds 7,165 files (7,372 = files + 207 folders;
  the enumeration also omitted `rate/`, the largest stage at 4,696 files)
  — yet the mechanical check reported "no file-count claims adjacent to HF
  tree links". The adjacency pattern does not reach this footer shape;
  only the Claude critic's manual re-count caught it (round-1 REVISE).
- **Why it is a workflow gap:** the check exists precisely to catch stale
  or wrong file-count claims; a claim shape that real bodies use (backtick
  sub-path + parenthetical count in a footer row with the pinned tree
  link) escaping the matcher is a coverage hole in the mechanical pass.
- **Confidence (emitter):** high (live miss with measured ground truth).

## Proposed change (candidate diff sketch — refine in planning)

```
+ verify_task_body.py, the HF file-count check:
+   - extend the claim regex: within a footer row containing a
+     huggingface.co/.../tree/<sha> link, match `sub/path` (backticked)
+     followed by "(<N> files" / "(<N> files:" parentheticals
+   - resolve the count via recursive scoped list_repo_tree(path_in_repo=
+     <sub-path>, revision=<sha>) counting RepoFile entries ONLY (folders
+     excluded — the #1112 miss counted folders)
+   - keep the existing network-fail-soft behavior (WARN, never FAIL, on
+     Hub unreachability)
+ tests: the #1112 footer shape as a fixture (wrong count -> FAIL row;
+   correct count -> PASS; folder-inflated count -> FAIL).
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep first: `grep -n 'file' scripts/verify_task_body.py | grep -i count`

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: d8e34254b667

Surfaced prose (verbatim, clean-result-critic round 1 on #1112):
"`scripts/verify_task_body.py`'s 'HF file-count claims match the Hub tree'
check reported no file-count claims adjacent to HF tree links while this
real '(7,372 files: …)' claim sat in the footer — the adjacency pattern
doesn't reach a parenthetical after a backtick sub-path within the same
footer row as the pinned `/tree/<sha>` link. Concrete change: widen the
claim-matcher to that shape and compare against a recursive scoped
`list_repo_tree` RepoFile count at the cited revision."
