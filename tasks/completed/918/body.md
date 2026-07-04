---
title: 'workflow-fix: gotchas.md — HF data-repo staging must use scoped list_repo_tree,
  not snapshot_download'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f158cca4aa4b
created_at: '2026-07-03T08:26:28Z'
has_clean_result: false
origin_prompt: 'failure-lesson #833 r7a: snapshot_download enumerates full ~1M-file
  tree before allow_patterns; scoped list_repo_tree + per-file pool is the fix (22388e4b3d)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson
(`gotcha_candidate: yes`) raised on task #833 (emitting agent: orchestrator,
round 7a GCP staging).

## Goal

Add a `.claude/rules/gotchas.md` bullet: staging any subtree of the huge
project data repo must use SCOPED `list_repo_tree(path_in_repo=...)` + a small
`hf_hub_download` thread pool — never `snapshot_download` + `allow_patterns`
(full-tree enumeration) and no longer bare `list_repo_files` (times out).

## Workflow gap

- **Bug observed:** a GCE staging step wedged 40+ min at 4.6% CPU with ZERO
  files landed: `snapshot_download` enumerated the entire ~1M-file
  `superkaiba1/explore-persona-space-data` tree before `allow_patterns`
  filtered (#833 round 7a, 2026-07-03; fix commit 22388e4b3d).
- **Why it is a workflow gap:** gotchas.md carries the known HF traps
  (list-truncation #375/#399, xet-CDN #515) but not this one; the existing
  agent-memory prescription ("use list_repo_files") is itself stale —
  `list_repo_files` on this repo now exceeds 90 s. Any future driver staging
  from the data repo re-hits this.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **HF data-repo staging (huge repo):** `snapshot_download(...,
+   allow_patterns=...)` enumerates the ENTIRE ~1M-file tree before
+   filtering (unbounded under the 2500 req/5min quota; #833 r7a wedged
+   40+ min, 0 files), and bare `list_repo_files` times out. Stage with
+   scoped `list_repo_tree(path_in_repo=<prefix>, recursive=True)` +
+   `hf_hub_download` in a ≤6-worker pool with backoff (ONE process; 9
+   concurrent processes tripped the quota in #833 r3). Recipe:
+   `scripts/issue833_gcp_phase_d.sh` (22388e4b3d); agent-memory twin:
+   experiment-implementer/feedback_hf_snapshot_download_full_tree_enumeration.md
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'snapshot_download' .claude/ CLAUDE.md scripts/`) and update
  every stale prescription; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: f158cca4aa4b

epm:failure-lesson v1 on #833 (2026-07-03, round 7a): snapshot_download
full-tree enumeration trap; root_cause_confirmed: yes; gotcha_candidate: yes.
