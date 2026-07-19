---
title: 'workflow-fix: lint upload_file-in-loop in issue drivers'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dca3e9dc2482
- daily-auto-filed
created_at: '2026-07-19T07:08:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): #1481''s driver shipped
  a per-file upload_file loop (~1,400 commits) causing an HF 429 storm; workflow_lint
  has no check flagging upload_file inside a loop despite the prose gotchas anti-pattern
  (c3-P13).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-18 /daily
transcript problem sweep (c3-P13). Route-2 filing.

## Goal

Add a `workflow_lint.py` check flagging `upload_file` calls inside a loop in
`scripts/issue*` driver scripts (WARN/FAIL) — the per-file upload loop is the
known 429-storm anti-pattern the gotchas rule already warns against in prose.

## Workflow gap

- **Bug observed:** #1481's driver shipped a per-file `upload_file` loop
  (~1,400 commits planned), causing an HF 429 storm; the known anti-pattern
  ("use one bulk `upload_folder` commit, never a per-file `upload_file` loop")
  reached production code despite plan + impl review.
- **Why it is a workflow gap:** `workflow_lint.py` has upload checks
  (`check_upload_or_true` swallow detection, `check_upload_as_file`,
  `check_upload_prefix_clobber`) but NONE flags an `upload_file` call INSIDE a
  loop. The prose gotchas rule is docs-only with no mechanical gate.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -cn 'upload_file.*loop\|loop.*upload_file\|per-file upload' scripts/workflow_lint.py` → 4 hits, all in the `check_upload_or_true` doc-comment describing the swallow anti-pattern (no loop-detection check); `grep -n 'def check_' scripts/workflow_lint.py | grep -i upload` → check_upload_or_true / check_upload_as_file / check_upload_prefix_clobber only — none detects an upload_file-in-loop shape (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
# workflow_lint.py: add check_upload_file_in_loop() —
+ AST-scan scripts/issue*.py (and scripts/issue_*/*.py) for an ast.Call to
+ upload_file / api.upload_file whose enclosing scope is an ast.For/ast.While
+ (or a comprehension); FLAG it (per-file upload loop 429-storm anti-pattern —
+ use a single bulk upload_folder commit). Waiver comment
+ (# UPLOAD_LOOP_EXEMPT: <reason>) for a genuinely bounded/small loop.
+ Bundle into the no-flags default run; add a fixture test (fails on a loop,
+ passes on bulk upload_folder + on a waived loop).
```

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Add the check + register it in the no-flags default; add
  `tests/test_workflow_lint.py` fixtures.

## Constraints / invariants

- Workflow-surface only. AST-based (not a fragile line regex); fail toward
  false-negative (bounded loops waivable) to keep the pre-commit gate quiet.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies.

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: 8dc815a6a0c4

Surfaced problem (c3-P13): #1481's driver ran a per-file upload_file loop
(~1,400 commits) → HF 429 storm; no lint caught the anti-pattern.
