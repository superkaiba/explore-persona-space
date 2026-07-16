---
title: 'daily-fix: sub-floor sentinel triggers apply run'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6a8bf229bb8e
- daily-auto-filed
created_at: '2026-07-16T07:20:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): 13:31Z vm-disk-low CRITICAL
  (15 GiB) preceded the 23:00Z hard-full by ~9.5h with only sidecar rows; / hit 100%
  3x on 07-15 wedging commits, worktrees, Bash, Step 9c'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

When VM-root free space drops below EPM_VM_DISK_SUBFLOOR_GIB, the watcher's sub-floor sentinel triggers an immediate `vm_disk_guard.py --apply` reclaim run (repeating on a short interval while below floor), instead of only writing a sidecar row.

## Workflow gap

- **Bug observed:** the 13:31Z vm-disk-low CRITICAL (15 GiB free) preceded the 23:00Z hard-full by ~9.5 h with only sidecar rows in between; `/` then hit 100% and blocked a git commit (#1366), worktree creation (#1367, #1359, #1361), wedged Bash + the Step 9c gate 4× (#1363), and dropped bg output (#779). Root disk hit 100% at least 3 separate times on 07-15.
- **Why it is a workflow gap:** the sub-floor sentinel is documented as "faster re-check intent, NOT remediation" (autonomous_session_watch.py L3009-3011) — so the fleet has a 9.5-h window in which the pressure is attributed but nothing reclaims until the hourly guard threshold path fires or the disk is already full.
- **Severity:** high
- verified-at-filing: `grep -n 'sub-floor' scripts/autonomous_session_watch.py` → L3009-3011 "The sub-floor sentinel is an ... faster re-check intent, NOT remediation: it writes a `band=sub-floor` row" (presence of the sentinel + absence of any reclaim trigger confirmed); `decide_subfloor` L3205 / `decide_subfloor_pct` L3318 / `vm_disk_pass` L19318 contain no `vm_disk_guard.py --apply` invocation (`grep -n 'vm_disk_guard' scripts/autonomous_session_watch.py` in the sub-floor region → none) (2026-07-16 UTC).

## Proposed change (refine in planning)

In `scripts/autonomous_session_watch.py`'s vm-disk pass (`vm_disk_pass` L19318, sub-floor sentinel around L3009-3220): when free bytes < `EPM_VM_DISK_SUBFLOOR_GIB`, additionally invoke `scripts/vm_disk_guard.py --apply` immediately (subject to the guard's own tier safety contract — terminal-status caches only, `store/`/`eval_results/` never touched), and repeat on a short interval while below floor. Keep the existing sidecar row + attribution unchanged. Note the structural fix (#681 cutover) is tracked separately as open task #1038 — this is the interim remediation leg.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (sub-floor sentinel, L3009-3220; `vm_disk_pass` L19318)
- Secondary: `scripts/vm_disk_guard.py` (invoked; may need a fast-path entry)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Never auto-delete an ACTIVE task's data (#679 warn-only invariant for active caches is preserved — the --apply run only reaps terminal-status caches).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 6a8bf229bb8e

- workflow_fix_target: scripts/autonomous_session_watch.py

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: batches 00 P1/P14, 02 P3, 04 P12, 09 P12 — sessions 4e46bb28, 5464a16a, 272c80a1, ada8210a, 3b499fa0, 3817ef84, c7b67f30. (Structural fix = #681 cutover, tracked as existing task #1038 — see no-action.)
