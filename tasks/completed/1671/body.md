---
title: 'daily-fix: sync_repo_root retries abort on HEAD.lock race'
kind: infra
tags:
- wf-fix
- wf-fix-fp:665aa314a8e4
- daily-auto-filed
created_at: '2026-07-25T06:48:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): During the 1645 session
  sync_repo_root exited 5 then 6 with rebase --abort rc=128 ''Unable to create .git/HEAD.lock:
  File exists'' (race with a concurrent git writer), leaving the SHARED repo root
  detached mid-rebase with 40+ unpushed commits until manual recovery'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session 5349b9f5, task #1645).

## Goal

`sync_repo_root.py` must not exit leaving the shared repo root detached mid-rebase when its `rebase --abort` loses a transient lock race.

## Workflow gap

- **Bug observed:** during #1645's Step 10d window the helper's own recovery failed twice — exit=5 "young rebase-merge husk (123s old)" then exit=6 with `['git','-C',...,'rebase','--abort'] rc=128 | stderr: error: Unable to create '.git/HEAD.lock': File exists` (2 distinct error exits in one guard run). The shared root sat detached mid-rebase with 40+ unpushed commits until the session manually verified the lock was gone, ran `rebase --abort` (ABORT-OK), and re-ran sync → `state=synced exit=0 ... 41 ahead pushed`. A second same-day near-miss: #1648's session hit a detached-root refusal from task.py during the same divergence window.
- **Why it is a workflow gap:** the helper exists precisely to make concurrent-root recovery safe; an abort that dies on a transient `HEAD.lock` (a concurrent git writer) is the same transient class as the documented index.lock bounded-poll rule, but the helper treats it as terminal.
- **Confidence (emitter):** high.
- verified-at-filing: `grep -n 'HEAD.lock' scripts/sync_repo_root.py` → 0 hits (no lock-poll/retry exists — absence bind; semantic sibling probe: abort paths at :672-:673/:741/:873 have no rc=128 lock retry); `git log --oneline --since='7 days ago' -- scripts/sync_repo_root.py` → only 993f0ccb78 (#1525 conflict-abort defusal recipe), no lock-retry fix landed (2026-07-25).

## Proposed change (candidate diff sketch — refine in planning)

In the husk-abort path: on `rebase --abort` rc=128 whose stderr matches `HEAD\.lock.*File exists` (and sibling ref-lock shapes), bounded-poll the lock file (~60s) and retry the abort once or twice before exiting nonzero; log each retry.

## Scope / surfaces

- Primary target: `scripts/sync_repo_root.py`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 665aa314a8e4

- workflow_fix_target: scripts/sync_repo_root.py
