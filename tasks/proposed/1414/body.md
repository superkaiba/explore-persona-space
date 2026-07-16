---
title: 'daily-fix: plan 9 pins out-root filesystem mount'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bb3280020548
- daily-auto-filed
created_at: '2026-07-16T07:22:51Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): #1333 attempt-3 crashed
  ENOSPC during checkpoint serialization because the out-root landed on the wrong
  (small) filesystem despite a section-9 GB estimate'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Plan §9 disk sizing names the target filesystem/mount for each out-root (not just the GB number), and workload preambles assert headroom against that mount (the #1333 `_assert_out_root_headroom` pattern, generalized).

## Workflow gap

- **Bug observed:** #1333 attempt-3 crashed with "No space left on device" during checkpoint serialization because the out-root landed on the wrong (small) filesystem despite a §9 GB estimate; fixed in-session by anchoring the out-root under /workspace + headroom probes (fc2b61b7 14:39-14:41Z).
- **Why it is a workflow gap:** plan-compute-sizing.md sizes disk in GB but never binds the estimate to a FILESYSTEM/MOUNT — a correct GB number on the wrong mount still crashes, and no preamble contract asserts headroom against the mount the out-root actually resolves to.
- **Severity:** low
- verified-at-filing: `grep -n 'mount\|filesystem\|out-root\|out_root' .claude/rules/plan-compute-sizing.md` → 0 hits for mount/filesystem/out-root binding (existing "headroom" hits at L84/L298/L313 cover Hub storage, the #1010 RAM/disk flag gate, and earlyoom RAM — none name a target mount for disk out-roots) — proposed leg absent (2026-07-16 UTC).

## Proposed change (refine in planning)

Extend `.claude/rules/plan-compute-sizing.md`'s disk-sizing section: every §9 disk estimate for an out-root (checkpoints, stores, staged inputs) NAMES the target filesystem/mount it lands on (e.g. `/workspace` volume on RunPod, the boot disk on GCE, the `/mnt/eps-data` bind on the VM), and the workload preamble asserts headroom against THAT mount before the write-heavy phase (generalize #1333's `_assert_out_root_headroom`: resolve the out-root path's mount via `os.statvfs` and fail loud below the sized headroom). Add the corresponding one-line expectation to planner §9 guidance if the rule's enforcement pointer needs it.

## Scope / surfaces

- Primary target: `.claude/rules/plan-compute-sizing.md`
- Secondary: `.claude/agents/planner.md` §9 (naming the mount per out-root), possibly a shared preamble helper for the `_assert_out_root_headroom` pattern

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: bb3280020548

- workflow_fix_target: .claude/rules/plan-compute-sizing.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: fc2b61b7 (#1333) 14:39-14:41Z (batch 05 P6).
