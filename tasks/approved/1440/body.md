---
title: 'daily-fix: task.py WARN on v<k>-stamped note heads'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f249739bb07c
- daily-auto-filed
created_at: '2026-07-17T06:51:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the poster-side _looks_field_led
  advisory WARN (scripts/task.py L137/L602) does not flag a ''v<k>. ''-stamped note
  head, so emitters keep echoing marker-version grammar into note heads; #1382 shipped
  only the read-side parser tolerance'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from a parked prose candidate on task #1382 (Alternatives critic, plan round 1), sub-candidate (2).

## Goal

Warn emitters at post time when a marker note head carries a 'v<k>. ' version stamp, preventing the emission class the #1382 parser tolerance absorbs.

## Workflow gap

- **Bug observed:** the poster-side _looks_field_led advisory WARN (scripts/task.py L137/L602) does not flag a 'v<k>. '-stamped note head, so emitters keep echoing marker-version grammar into note heads; #1382 shipped only the read-side parser tolerance
- **Why it is a workflow gap:** Advisory WARNs at the mutation entrypoint are the project's mechanism for steering emitters; a WARN blind to a known bad shape leaves the class recurring.
- **Confidence (emitter):** low (emitter) — concrete file + change, filed per the 2026-06-11 standing directive
- verified-at-filing: `grep -n '_looks_field_led' scripts/task.py` -> 2 hits (def at L137, call at L602); no 'v<k>'/version-stamp pattern in the predicate (absence claim; semantic probe: the predicate checks field-led note shapes only)

## Proposed change (candidate diff sketch — refine in planning)

Extend _looks_field_led (or add a sibling advisory check at the same L602 call site) to match a leading r'v\d+\.\s' note head and print the advisory WARN naming the stamp shape.

## Scope / surfaces

- Primary target: `scripts/task.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: f249739bb07c



