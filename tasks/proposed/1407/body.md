---
title: 'daily-fix: Monitor keys on exit/mtime not existence'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c599fb5f7120
- daily-auto-filed
created_at: '2026-07-16T07:22:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): A Monitor fired ''done''
  instantly on a stale pre-existing v1 JSON instead of waiting for the v2 re-run (#779)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

One-line addition to CLAUDE.md § Monitoring: when re-running a script whose output path already exists, the monitor condition keys on process exit or a fresh mtime, never bare file existence.

## Workflow gap

- **Bug observed:** a Monitor fired "done" instantly on a stale pre-existing v1 JSON instead of waiting for the v2 re-run (8b076180, #779, 05:50Z) — the stale-EXISTING-file sibling of the #825 empty-dir false-DONE already covered by the ownership-check bullet.
- **Why it is a workflow gap:** the monitoring rules (§ Monitoring; the ownership-check bullet's verify-first trigger) cover the empty-dir false-DONE but nothing bans existence-keyed monitor conditions on RE-RUNS, so a stale prior-version output satisfies the condition immediately.
- **Severity:** medium
- verified-at-filing: `grep -n '## Monitoring' CLAUDE.md` → L460 (target section exists); `grep -n 'mtime' CLAUDE.md` → 1 hit (L172, the unrelated non-canonical-cache recency gate) — no monitor-condition mtime/exit rule anywhere in CLAUDE.md (absence confirmed) (2026-07-16 UTC).

## Proposed change (refine in planning)

Append one bullet to `CLAUDE.md` § Monitoring (L460): "When re-running a script whose output path already exists, key the Monitor/until-loop condition on process exit or a fresh output mtime (newer than the launch timestamp), never bare file existence — a stale prior-version artifact satisfies an existence check instantly (#779, 2026-07-15; sibling of the #825 empty-dir false-DONE)."

## Scope / surfaces

- Primary target: `CLAUDE.md` (§ Monitoring, L460)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: c599fb5f7120

- workflow_fix_target: CLAUDE.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 8b076180 (#779) 05:50Z (batch 06 P7).
