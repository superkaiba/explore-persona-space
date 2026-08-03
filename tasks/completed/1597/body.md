---
title: 'daily-fix: rotate inner append-mode logs on relaunch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bfc366002613
- daily-auto-filed
created_at: '2026-07-22T06:47:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): inner append-mode per-cell
  train.log not rotated on relaunch; crash-signature grep false-fired ASSERT-FAILED-AGAIN
  on the previous crash''s lines'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-21 from the #1112 false "ASSERT-FAILED-AGAIN" monitor alarm (transcript 24ae2158).

## Goal

Extend the pod-side relaunch log-rotation duty (rule 1b) to INNER per-cell logs opened in append mode — or require crash-signature greps to be scoped past the launch timestamp.

## Workflow gap

- **Bug observed:** after the #1112 fix-4 relaunch, the health monitor raised "ASSERT-FAILED-AGAIN" by grepping the UNROTATED, append-mode inner `train.log` — counting the PREVIOUS crash's assert lines as fresh failures (~1 min triage; could have triggered a wrong kill/relaunch).
- **Why it is a workflow gap:** `.claude/rules/pod-side-reporting.md` rule 1b mandates rotating "the phase log" at every relaunch, but the duty as written covers the dispatcher-level log the pattern poller scans — the inner per-cell `train.log` a dispatcher subprocess appends to is not named, and crash-signature greps against it are launch-timestamp-unsafe (sibling of the #952 un-rotated-log false-fire and the #779 stale-artifact false-DONE class).
- **Confidence:** medium.
- verified-at-filing: `sed -n '225,245p' .claude/rules/pod-side-reporting.md` → rule 1b present, scoped to "the phase log"/predecessor's log; no inner/per-cell append-mode extension (in-target presence + absence probe, 2026-07-22). `git log --oneline --since='7 days ago' -- .claude/rules/pod-side-reporting.md` → no such extension landed.

## Proposed change (candidate diff sketch — refine in planning)

In rule 1b (or a sibling 1c): the rotation duty covers EVERY log a crash-signature grep may scan — including inner per-cell logs opened append-mode by dispatcher subprocesses; where rotation is impractical, the grep must be scoped past the launch timestamp (`awk`/byte-offset from a launch sentinel), never a whole-file pattern match.

## Scope / surfaces

- Primary target: `.claude/rules/pod-side-reporting.md`

## Constraints / invariants

- Workflow-surface only; consistent with rule 1b's existing #952 rationale.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: bfc366002613

- workflow_fix_target: .claude/rules/pod-side-reporting.md

Origin evidence: transcript 24ae2158, 2026-07-21T07:21:45Z (monitor "ASSERT-FAILED-AGAIN") vs 07:22:06Z ("False alarm ... stale content in the *unrotated inner* train.log from the previous crash (append mode)").
