---
title: 'daily-fix: step9c oracle detached from root dirt'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8d992e55716e
- daily-auto-filed
created_at: '2026-07-16T07:22:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): #1317''s baseline compare
  serialized ~3h20m behind #825''s live workload dirtying the shared root despite
  #1251; #1363''s gate needed fixture temp writes moved to the data disk'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Run the Step 9c pristine-oracle baseline compare in a detached scratch worktree at origin/main by default, so other sessions' shared-root dirt never serializes an infra gate; default step9c fixture TMPDIR onto the data disk.

## Workflow gap

- **Bug observed:** #1317's baseline compare serialized ~3h20m behind #825's live GCP workload dirtying the shared root (5 bounded poll legs; 0f9a8fbf 08:38-11:54Z) — despite #1251 ("step9c baseline oracle tolerates unrelated concurrent dirty", completed) the overlapping-dirt case still blocks; separately #1363's gate needed fixture temp writes redirected to the data disk to pass under disk pressure (5464a16a 22:37-23:04Z).
- **Why it is a workflow gap:** the scratch-worktree machinery exists in step9c_baseline.py but only as an eligibility-gated FALLBACK — a live sibling workload continuously re-dirtying the root keeps the primary path waiting instead of routing straight to the detached oracle, and fixture TMPDIR defaults to `/` under disk pressure.
- **Severity:** medium
- verified-at-filing: `grep -n 'scratch\|dirty\|TMPDIR' scripts/step9c_baseline.py` → scratch machinery PRESENT as fallback (`--scratch-timeout-s` L26, `--no-scratch-fallback` L27, "scratch-worktree fallback creation" L53, scratch PYTHONPATH shadow L61-70; "dirty oracle on a failing node where the scratch fallback is ineligible" L46-49) — the eligibility-gated fallback exists but the 07-15 #1317 serialization post-dates it, showing the overlapping-dirt case is not covered; no TMPDIR/data-disk routing hit (absence confirmed for that leg) (2026-07-16 UTC).

## Proposed change (refine in planning)

In `scripts/step9c_baseline.py` + `.claude/skills/issue/SKILL.md` Step 9c: (a) make the detached scratch worktree at origin/main the DEFAULT oracle root (or auto-route to it as soon as the shared root reads dirty once, rather than polling the root clean across bounded legs) so a concurrent session's continuous dirt never serializes the gate — reconcile with #1251's tolerance work and the existing eligibility gates (L46-53) to see which predicate kept #1317 on the waiting path; (b) default step9c fixture temp writes (TMPDIR) onto the data disk (the #1363 in-session fix, generalized) so the gate passes under root-disk pressure.

## Scope / surfaces

- Primary target: `scripts/step9c_baseline.py` (scratch fallback, L26-70)
- Secondary: `.claude/skills/issue/SKILL.md` Step 9c (gate invocation)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The pristine-oracle correctness contract (HEAD-pinned scratch, PYTHONPATH shadow, never the invoking sys.executable) is preserved.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 8d992e55716e

- workflow_fix_target: scripts/step9c_baseline.py

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 0f9a8fbf (#1317) 08:38-11:54Z (batch 09 P3); 5464a16a (#1363) 22:37-23:04Z (batch 00 P14).
