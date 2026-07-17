---
title: 'daily-fix: triage observer empty-window false flags'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a14b4b34f94b
- daily-auto-filed
created_at: '2026-07-16T07:21:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): Observer posted 2 violation
  flags on #1005 dispatches whose re-enumerated window had 0 candidates, with a preceding
  ''external-markers triaged: none'' note ~2 min earlier'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Suppress the triage-observer flag when the re-enumerated candidate window is empty, and accept a preceding same-window `external-markers triaged:` progress note as satisfying the triage duty.

## Workflow gap

- **Bug observed:** the observer posted 2 violation flags on #1005 dispatches (epm:progress 19:33:01Z + 20:13:01Z) whose re-enumerated candidate window had 0 candidates, and where an "external-markers triaged: none" note preceded the dispatch by ~2 min (session notes 18:51:16Z / 19:39:44Z) — contradicting the pass's own documented non-empty-window predicate.
- **Why it is a workflow gap:** the observer's documented contract is to flag "a missing / 'none' triage line against a NON-EMPTY candidate set", so a flag on a 0-candidate window (or where a preceding same-window triage note exists) is a false positive that erodes trust in the sidecar/push channel.
- **Severity:** medium
- verified-at-filing: `grep -n 'non-empty' scripts/autonomous_session_watch.py` → docstring hits at L366-367 and L4365 ("flags a missing / 'none' triage line against a non-empty candidate set") — the documented predicate is confirmed present; the flag path emits violation kind `none-with-candidates` (L4495) with a `candidate_count` field (L4502/L4614/L4641), yet the 07-15 flags fired with candidate_count context on windows the miner re-enumerated as EMPTY — the code/predicate divergence (window semantics or the preceding-note adjacency at `decide_triage_observer_actions` L4511) is the bug to localize; behavioral claim from transcript+events evidence, target section existence verified (2026-07-16 UTC).

## Proposed change (refine in planning)

In `scripts/autonomous_session_watch.py`'s triage-observer pass (block starting L4358; `decide_triage_observer_actions` L4511, `triage_observer_pass` L4655): (a) hard-suppress any flag whose re-enumerated window has 0 candidates — enforcing the documented "non-empty" predicate in code; (b) treat a preceding `external-markers triaged:` epm:progress note within the same adjacency window (even one reading "none") as satisfying the duty for that dispatch. Reconcile the observer's window re-enumeration with the #889 enumerator semantics the session itself used, since the 07-15 divergence suggests the two enumerate different windows.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (triage-observer pass, L4358-4655)
- Secondary: `task_workflow.audit_dispatch_triage` (the shared enumerator, if the window semantics live there)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The pass stays observe/alert-only (never mutates status / stops sessions / blocks dispatches — pinned by tests).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: a14b4b34f94b

- workflow_fix_target: scripts/autonomous_session_watch.py

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: #1005 events epm:progress 19:33:01Z + 20:13:01Z vs session notes 18:51:16Z / 19:39:44Z (batch 01 P12).
