---
title: 'daily-fix: pid file from launch chain child, never pgrep'
kind: infra
tags:
- wf-fix
- wf-fix-fp:200a28402d99
- daily-auto-filed
created_at: '2026-07-23T07:03:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): a relaunch populated the
  pid file from a quick pgrep that captured a TRANSIENT pid — 2 false exited alarms
  (#1112); item 1 says how to WRITE the pid but not how to ACQUIRE it'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). In #1112 (24ae2158, 2026-07-23T05:52–06:30Z), the Monitor TWICE reported the pod dispatcher "exited" while it was alive: the launch chain had populated the pid file from a quick `pgrep` that captured a TRANSIENT pid. Fixed in-session by writing the python child pid and probing the process PATTERN — but the pid-file contract does not warn against the pgrep-capture shape.

## Goal

`pod-side-reporting.md` item 1 gains the missing clause: the pid file is populated from the LAUNCH EXPRESSION itself (`$!` of the setsid child / the launcher's `echo $$`), NEVER from a post-hoc `pgrep` (which can capture a transient); and liveness monitors probe the process PATTERN (bracketed per the self-match gotcha) as a fallback alongside the pid-file.

## Workflow gap

- **Bug observed:** 2 false "exited" Monitor alarms in one relaunch sequence (24ae2158), each burning a diagnosis round during an already-fragile ZeRO-3 recovery.
- **Why it is a workflow gap:** item 1 prescribes WRITING the new pid atomically and confirming it, but says nothing about HOW to obtain the pid when relaunching without the launcher — the observed failure used `pgrep` and captured a transient.
- **Confidence:** high.
- verified-at-filing: `.claude/rules/pod-side-reporting.md` item 1 (lines 207–232) read in full at filing: it covers the `echo $$` launcher path and the explicit `$CHILD_PID` atomic rewrite, with NO clause on pid ACQUISITION (pgrep-capture ban) nor the pattern-probe fallback (absence claim, context read), 2026-07-23 UTC.

## Proposed change (refine in planning)

Add a sibling clause 1d (or extend 1): pid acquisition comes from the launch expression (`$!` / launcher `$$`), never a post-hoc pgrep; monitors probe the bracketed process pattern in addition to the pid-file.

## Scope / surfaces

- Primary target: `.claude/rules/pod-side-reporting.md` (item 1 family).

## Constraints / invariants

- The #1597 log-rotation clauses (1b/1c) unchanged. Recursion guard applies.

## Provenance

- fingerprint: 200a28402d99

- workflow_fix_target: .claude/rules/pod-side-reporting.md
