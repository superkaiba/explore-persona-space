---
title: 'daily-fix: sweep subsumption by fix merge time not creation'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d273d3de43f1
- daily-auto-filed
created_at: '2026-07-22T06:47:48Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): sweep suppression keys
  on fix-task creation time so a candidate parked before the fix merged is enumerated
  as a genuine re-raise'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-21 from the #1577/#1579 sibling-blindness incident (transcripts 13f0b90c / 2e1f1383) — and validated by THIS run's own Step C sweep, which enumerated the stale #1577 candidate and required a manual dedup.

## Goal

Extend the parked-candidate sweep's subsumption rule so a parked candidate whose `target_file` matches a fix task MERGED/closed AFTER the candidate was parked is suppressed (compare against merge/close time, not only task-creation time).

## Workflow gap

- **Bug observed:** #1577's session parked a candidate at 10:59:07Z naming `scripts/select_step9c_tests.py` ("stem/literal/--map-files arms are .py-only"); #1579's session merged the general fix for that exact bug (PR #1356, `f0770307ce`) at 11:00:11Z — 64 seconds later. Tonight's Step C sweep enumerated the candidate anyway (it postdates #1579's CREATION), and only a manual verified-at-filing grep against the current tree prevented a duplicate pipeline session (the #1479/#1476 class, ~2.3h burned last occurrence).
- **Why it is a workflow gap:** `scripts/sweep_parked_wf_candidates.py`'s suppression rule for TERMINAL fix-task hits keys on the fix task's creation time ("a candidate that predates the closed fix was subsumed by it; one raised after it closed is a genuine re-raise") — a candidate parked after the fix task's creation but before its MERGE reads as a genuine re-raise, when the merged fix already covers it. Additionally the recursion-guard park path runs no sibling-overlap advisory at park time.
- **Confidence:** high (reproduced live tonight).
- verified-at-filing: `grep -n 'creation\|suppress' scripts/sweep_parked_wf_candidates.py` → the creation-time predicate is present at ~:47-50 ("suppresses only when the task's creation ... predates") with no merge-time comparison (presence + absence probes bind, 2026-07-22); `git log --oneline --since='7 days ago' -- scripts/sweep_parked_wf_candidates.py` → no merge-time extension landed.

## Proposed change (candidate diff sketch — refine in planning)

In `sweep_parked_wf_candidates.py`: for a TERMINAL (completed/archived) same-file fix-task hit, suppress when the candidate's park ts predates the fix task's MERGE/close time (read `epm:merged` / the terminal `epm:status-changed` ts from the fix task's events), not only its creation time; keep fp-exact suppression unchanged; optionally emit the suppressed row with `suppressed_by.kind: "merged-after-park"` for auditability. Secondary (planner's call): have the recursion-guard park path print the `open_workflow_fix_siblings` advisory at park time.

## Scope / surfaces

- Primary target: `scripts/sweep_parked_wf_candidates.py`
- Siblings: `.claude/rules/workflow-fix-on-bug.md` § Recursion guard (doc), its tests.

## Constraints / invariants

- Fail-open toward enumeration on unreadable events (a wrongly-suppressed candidate is a dropped bug; a wrongly-enumerated one costs a manual dedup — keep the asymmetry).
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: d273d3de43f1

- workflow_fix_target: scripts/sweep_parked_wf_candidates.py

Origin evidence: transcript 13f0b90c (park marker 2026-07-21T10:59:02-07Z) vs PR #1356 merge `f0770307ce` at 11:00:11Z; tonight's /daily Step C output enumerated the candidate with `suppressed: false`, `open_wf_fix_on_file: 865` (a different-bug advisory hit).
