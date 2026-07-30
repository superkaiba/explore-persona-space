---
title: 'daily-fix: remote-landing watches need fence deadline + hear'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e08121f743c1
- daily-auto-filed
created_at: '2026-07-30T07:00:33Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): Two sessions went dark
  for 3.5h / 58min behind silent bg watches: an until-loop keyed on an HF landing
  that never came (producer powered off), and a Monitor poll chain that never woke
  again — the watcher stall-alert was the only recovery both times'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miners B-P1 (session 77ee3ced, #1738, ~3.5h dark) and C-P3 (session bae92cbd, #1739, ~58min dark)).

## Goal

A bg watch keyed on a remote landing must wake the session by its producer's fence even when the landing never happens; a long-armed silent Monitor must be distinguishable from a dead one.

## Workflow gap

- **Bug observed:** #1738: an until-loop keyed on an HF chunk landing ran silently past the producing GCE instance's ~15:08Z poweroff; no assistant turn 14:32->17:53 until the watcher respawned. #1739: after one healthy Monitor wake the session idled; no wake again; watcher flagged ALIVE-BUT-STALLED at 08:03 and respawned at 08:23 — ~58 min with no lane monitoring on a 3-lane GCP run.
- **Why it is a workflow gap:** the monitoring guidance sanctions until-loops/Monitor as the pollers but does not require a producer-fence deadline or a heartbeat, so both failure modes read as healthy idle to the session and near-dead to the watcher.
- **Confidence (emitter):** medium
- verified-at-filing: task markers probed by the miners: #1738 stalled-alert 17:23:44Z + respawn 17:43:41Z; #1739 stalled-alert 08:03:24Z + respawn 08:23:27Z + lanes-resume 08:27:48Z (task.py view, 2026-07-30).

## Proposed change (refine in planning)

In the SKILL's monitoring/detached-phase sections: (a) any until-loop keyed on a REMOTE artifact landing carries a deadline = producer fence + grace, exiting DEADLINE so the session wakes and re-checks the producer; (b) long-interval Monitor loops emit a no-op heartbeat every 2-3 cycles.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: e08121f743c1
