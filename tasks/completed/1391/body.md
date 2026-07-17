---
title: 'daily-fix: watcher re-drive w/o registration file'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bdaaaffba801
- daily-auto-filed
created_at: '2026-07-16T07:20:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): #1090 fu5 session died
  with its crash-recovery registration file gone; the watcher was blind and the ~55-min
  stall surfaced only via a manual ask'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Make the watcher re-drive an ACTIVE-status (or followups_running) task whose issue-mapped session is provably dead even when the crash-recovery registration file `~/.eps-autonomous/issue-<N>.json` is missing, inferring ownership from recent stage-dispatch markers / spawn history.

## Workflow gap

- **Bug observed:** #1090 fu5's session died post-smoke with its crash-recovery registration file GONE — the watcher's crash-recovery pass keys on the registry file, so recovery was fully blind; the ~55-min stall surfaced only because Thomas manually asked "is everything running" (194f5813 PM 07:22:15Z: "the crash-recovery registration is gone (so the watcher won't auto-recover it), and no pod ever launched").
- **Why it is a workflow gap:** the registration-independent orphan sweep exists but did not cover this case (its ~90-min ORPHAN_STALENESS floor and/or its follow-up-round semantics left the fu5 stall invisible for ~55 min until a human asked), so a dead session with a lost registration has no timely automated recovery.
- **Severity:** high
- verified-at-filing: `grep -n 'Orphan sweep' scripts/autonomous_session_watch.py` → present (docstring L113-131: registration-INDEPENDENT safety net, `decide_orphan` L11937 / `orphan_sweep_pass` L12223) — the generic mechanism exists but keys on `ORPHAN_STALENESS_S_DEFAULT` ~90 min and did not recover the 07-15 fu5 case; `grep -n 'stage-dispatch' scripts/autonomous_session_watch.py` → hits only in the followup-witness helper (L9105, `_FOLLOWUP_STAGE_DISPATCH_WITNESS_PREFIX`), not in orphan-recovery ownership inference — proposed inference leg absent (2026-07-16 UTC).

## Proposed change (refine in planning)

Extend `scripts/autonomous_session_watch.py`'s orphan/crash-recovery machinery so an ACTIVE-status or `followups_running` task whose issue-mapped session is provably dead (wrapper pid gone, transcript silent) is re-driven even when `~/.eps-autonomous/issue-<N>.json` is missing: infer session ownership from recent stage-dispatch breadcrumbs (`epm:progress` stage-dispatch notes, `_FOLLOWUP_STAGE_DISPATCH_WITNESS_PREFIX` at L9105) and spawn history, and evaluate whether the ~90-min orphan staleness floor should tighten for a provably-dead (vs merely unregistered) owner. The planner should reconcile with `decide_orphan` (L11937) so the existing cap/manual-registration guards are preserved.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Anchors: orphan sweep docstring L113-131; `decide_orphan` L11937; `orphan_sweep_pass` L12223; followup stage-dispatch witness L9105

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: bdaaaffba801

- workflow_fix_target: scripts/autonomous_session_watch.py

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 194f5813 (PM) 07:22:15Z: "the crash-recovery registration is gone (so the watcher won't auto-recover it), and no pod ever launched" (batch 09 P21).
