---
title: 'workflow-fix: gotchas — HF download-accel failure matrix (scope disables per
  workload)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:408752a68576
created_at: '2026-07-30T07:35:50Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed incident chain on #1739 (2026-07-30 05:52-07:06Z):
  xet wedge / hf_transfer error / plain >50GB refusal across three staging relaunches'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed incident chain on task #1739 (three consecutive staging crashes, 2026-07-30 05:52-07:06Z).

## Goal

Add a gotchas bullet documenting the HF Hub download-accelerator failure matrix: xet wedges indefinitely on many-small-file storms, hf_transfer fails without retry, and the plain path refuses >50GB-class files - so accelerator disables must be scoped per download workload, never process-wide on a mixed pipeline.

## Workflow gap

- **Bug observed:** Three consecutive staging crashes on task 1739's syc lane relaunches: xet_get wedge on a 458-small-file restore, hf_transfer RuntimeError on a tar download, and plain-path ValueError file-too-large on a 49GB store tar after a process-wide disable.
- **Why it is a workflow gap:** The fleet defaults (HF_XET_HIGH_PERFORMANCE=1 + HF_HUB_ENABLE_HF_TRANSFER=1, upload-policy) plus the existing per-launch override guidance imply the disables are a safe global switch; the incident chain shows the three transfer paths have DISJOINT failure domains (small-file storms vs large files), so a mixed pipeline (big tars + small-file restore in one process tree) needs PER-WORKLOAD scoping — a trap any future staging/restore script will re-hit, and gotchas.md (the on-demand codebase-trap registry) has no entry for the download side.
- **Confidence (emitter):** high
- verified-at-filing: `grep -in "xet\|hf_transfer" .claude/rules/gotchas.md` → hits cover the UPLOAD wedge ladder only (no download-matrix entry) (2026-07-30). Incident evidence on task #1739: epm:progress v76 (xet_get wedge, py-spy stack: huggingface_hub file_download.py:633), v77 (hf_transfer RuntimeError, file_download.py:485), epm:run-launched v8 (plain-path ValueError 'file too large ... Use hf_transfer or hf_xet', file_download.py:418, on the 49GB sycophancy_labeling tar); the landed scoped fix is commit 795f474712 on branch issue-1739 (scripts/issue1739_restore_partial.py sets HF_HUB_DISABLE_XET=1 + HF_HUB_ENABLE_HF_TRANSFER=0 before its hub import, callers keep accelerators for the big tars).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/rules/gotchas.md
+ - **HF Hub download-accelerator failure matrix — scope disables per
+   workload, never process-wide.** The three transfer paths fail in
+   DISJOINT domains: xet WEDGES indefinitely (no timeout around xet_get)
+   on many-small-file download storms (458x npz, #1739); hf_transfer
+   fails fast with a bare RuntimeError and no retry; the PLAIN path
+   refuses >50GB-class files (ValueError 'file too large'). A mixed
+   pipeline (multi-GB tars + a small-file restore) therefore cannot use
+   one global setting: keep accelerators ON for the big-file legs and
+   set HF_HUB_DISABLE_XET=1 + HF_HUB_ENABLE_HF_TRANSFER=0 INSIDE the
+   small-file-storm script before its huggingface_hub import (worked
+   impl: scripts/issue1739_restore_partial.py @ 795f474712).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md` (one bullet; sibling of the existing upload-side wedge-ladder entry in `.claude/rules/upload-policy.md` — cross-reference it, do not duplicate it).

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 408752a68576

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: Three consecutive staging crashes on task 1739's syc lane relaunches: xet_get wedge on a 458-small-file restore, hf_transfer RuntimeError on a tar download, and plain-path ValueError file-too-large on a 49GB store tar after a process-wide disable.
why_workflow_gap: The three HF transfer paths have disjoint failure domains; fleet-default accelerators + global-disable guidance leave the per-workload scoping rule undocumented in the trap registry.
proposed_change: Add a gotchas bullet documenting the download-accelerator failure matrix + the per-workload scoping recipe (worked impl 795f474712).
confidence: high
related_task: #1739
<!-- /workflow-fix-candidate -->
