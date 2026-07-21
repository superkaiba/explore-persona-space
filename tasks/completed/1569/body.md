---
title: 'daily-fix: probe free check id at implement time'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e6394ef1f568
- daily-auto-filed
created_at: '2026-07-20T06:48:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): two sessions added c40
  to verify_plan.py same day; PR conflict + renumber round'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from transcript-mined problems (see evidence in ## Provenance).

## Goal

Have implementers that add an ENUMERATED check id (verify_plan.py c<N>, verify_task_body.py check <N>) probe origin/main for the next free id at IMPLEMENT time (not plan time), avoiding same-day check-id collisions between concurrent workflow-fix sessions.

## Workflow gap

- **Bug observed:** two same-day workflow-fix sessions (#1550, #1551) independently added check id `c40` to scripts/verify_plan.py; the second PR (#1321) hit a genuine content conflict and needed a (fortunately pre-authorized) c40→c41 renumber + conflict-resolve round before merging.
- **Why it is a workflow gap:** check ids are assigned at plan time from a stale view of main; with 20+ concurrent wf-fix sessions/day the collision class recurs; the recovery worked but cost a bounce round.
- **Confidence (emitter):** low (recovery worked as designed; filed per the standing any-confidence directive)
- verified-at-filing: incident-anchored: session d825987e (task #1551) @ 20:32 UTC 2026-07-19, 'GraphQL: Pull Request has merge conflicts' → conflict in scripts/verify_plan.py + tests/test_verify_plan.py, both sides adding c40; renumber merged 20:51 (d6b652ba70 landed #1550's c40 first). No grep-refutable count claim made.

## Proposed change (candidate diff sketch — refine in planning)

(none — sketch: one sentence in the implementer spec: before committing a new enumerated check, `git fetch origin main` and grep the CURRENT max check id in the target file; renumber locally if taken)

## Scope / surfaces

- Primary target: `.claude/agents/implementer.md` (and/or `.claude/agents/experiment-implementer.md` if the pattern generalizes)

## Constraints / invariants

- Workflow-surface rules apply where the target is workflow surface; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies where tagged wf-fix (workflow_fix_target Provenance line below).

## Provenance

- sha-verify (filing-time, #1467): `d825987e` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: .claude/agents/implementer.md
- fingerprint: 10448fc0bdc0

Mined evidence: PR #1321 conflict + pre-authorized renumber round (#1551 vs #1550, 2026-07-19).
