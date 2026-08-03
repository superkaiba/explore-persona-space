---
title: 'daily-fix: route-1 fix closes the open task it credits'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d423591b6769
- daily-auto-filed
created_at: '2026-07-31T06:56:08Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): the same bug was routed
  BOTH as a /daily route-1 direct fix (commit crediting #1823) AND as filed+spawned
  task #1823; the spawned session booted only to discover mootness and archive itself.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; miner-2 P9, session 1ba4dcf9 / task #1823).

## Goal

When a /daily route-1 self-applied fix lands the exact deliverable of an already-filed task (the commit credits the task id), the /daily run closes/archives that task in the same pass instead of leaving its spawned session to discover mootness.

## Workflow gap

- **Bug observed:** the same E501 bug was routed BOTH as a /daily route-1 direct fix (commit 7202cfd669, subject crediting #1823) AND as filed+spawned task #1823; the spawned `/issue 1823` session booted, found the fix already on main, verified the test green, and archived itself — a wasted session spawn (~3 min, graceful but pure overhead).
- **Why it is a workflow gap:** the /daily three-route classifier has no cross-check between route-1 applied fixes and open filed tasks covering the same deliverable, so a double-routed bug costs a session spawn every time.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'crediting' .claude/skills/daily/SKILL.md` → 0 (no route-1-credits-open-task closing step exists; absence confirmed 2026-07-31 filing time).

## Proposed change (candidate diff sketch — refine in planning)

Add to the /daily route-1 apply recipe: after committing a route-1 fix, if the fix resolves an OPEN filed task (the problem's evidence names a task id, or the open-sibling scan matches), archive that task (`task.py set-status <id> archived` with a note naming the commit) and stop any spawned session — the same post-hoc remedy the closed-sibling advisory prescribes, applied at route time.

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md` (route-1 apply recipe / triage section)

## Constraints / invariants

- Only archive when the route-1 commit demonstrably covers the task's full deliverable (verify the task's failing test / acceptance criterion against the landed fix first); ambiguous → leave the task, note the possible overlap.
- Archiving is a status mutation — it must go through `task.py set-status` (canonical API), never a hand-edit.

## Provenance

- fingerprint: d423591b6769

- workflow_fix_target: .claude/skills/daily/SKILL.md
- origin: /daily 2026-07-30 miner-2 P9 (transcript 1ba4dcf9, #1823)
