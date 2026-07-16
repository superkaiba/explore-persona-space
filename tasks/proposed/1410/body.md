---
title: 'daily-fix: gotchas /mnt/eps-data root not writable'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f8dfa7081db3
- daily-auto-filed
created_at: '2026-07-16T07:22:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): Two sessions hit PermissionError
  [Errno 13] staging at the data-disk root (#1336 diag staging; #823 inline recovery
  mkdir /mnt/eps-data/tmp denied)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

One-line gotchas entry: the `/mnt/eps-data` root is root-owned and not directly writable — VM-local staging goes under the bind-mounted `.claude/worktrees/issue-<N>/data/` path (or a pre-created per-issue quota dir), never a fresh top-level `/mnt/eps-data/<dir>`.

## Workflow gap

- **Bug observed:** two sessions hit `PermissionError: [Errno 13] /mnt/eps-data/<dir>` staging at the data-disk root: #1336 diag staging (25ba019b 20:44:49Z) and #823's inline recovery `mkdir /mnt/eps-data/tmp` denied (b7150177 22:59:51Z).
- **Why it is a workflow gap:** nothing in the gotchas/disk-hygiene docs says the data-disk root is root-owned, so sessions under disk pressure improvise top-level `/mnt/eps-data/<dir>` staging and burn a recovery turn on the PermissionError.
- **Severity:** low
- verified-at-filing: `grep -n 'eps-data' .claude/rules/gotchas.md` → 0 hits — entry absent (2026-07-16 UTC).

## Proposed change (refine in planning)

Add one entry to `.claude/rules/gotchas.md`: "`/mnt/eps-data` root is root-owned — a direct `mkdir /mnt/eps-data/<dir>` fails `PermissionError: [Errno 13]`. VM-local staging goes under the bind-mounted `.claude/worktrees/issue-<N>/data/` path (per-issue ext4 quota, #681) or an already-created per-issue dir; never a fresh top-level `/mnt/eps-data/<dir>` (#1336, #823, 2026-07-15)."

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Companion: m06 (inline-analysis disk routing) points staging at the same worktree data path

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: f8dfa7081db3

- workflow_fix_target: .claude/rules/gotchas.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 25ba019b 20:44:49Z (batch 02 P4); b7150177 22:59:51Z (batch 01 P4).
