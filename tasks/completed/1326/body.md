---
title: 'daily-fix: Step-0 guard blind to inline-chat drivers'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dd7eb89d1131
- daily-auto-filed
created_at: '2026-07-15T06:52:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): a fresh /issue 952 session
  passed the Step-0 single-orchestrator guard while the live inline-chat session was
  driving the same-issue follow-up round (the inline driver is never in the spawn-session
  registry), dispatched the pod upload phase, and detected the collision only ~8 min
  in via concurrent marker posts while the other session was terminating pod-952'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep (session ae2aef04, #952, 05:13-05:25Z): duplicate-orchestrator near-miss — two orchestrators touched the same round/pod before the duplicate yielded cleanly (deliberate-stop reason=step0-session-collision, owner=inline-chat-session 28d0874a).

## Goal

close the Step-0 guard blind spot: an inline chat session taking a same-issue follow-up round registers itself in ~/.eps-autonomous/issue-<N>.json (or the Step-0 guard also treats a fresh stage-dispatch epm:progress by another writer as a live-driver signal) before dispatching work

## Workflow gap

- **Bug observed:** a fresh /issue 952 session passed the Step-0 single-orchestrator guard while the live inline-chat session was driving the same-issue follow-up round (the inline driver is never in the spawn-session registry), dispatched the pod upload phase, and detected the collision only ~8 min in via concurrent marker posts while the other session was terminating pod-952
- **Why it is a workflow gap:** the Step-0 single-orchestrator guard reads the spawn-session registry; the sanctioned inline-chat follow-up path (CLAUDE.md carve-out) creates a live driver the registry never sees, so the guard passes and a double-writer window opens on the pod + markers.
- **Confidence:** high (observed collision; yield was luck-of-ordering)
- verified-at-filing: n/a — behavioral gap; the Step-0 guard text names the spawn-session registry as its signal and no inline-registration duty exists in the carve-out or SKILL.md (grep "single-orchestrator guard" hits reference registry/live-session checks only) (2026-07-15).

## Proposed change

Either leg (planner picks): (a) the inline carve-out adds a register-current duty (`spawn_session.py register-current` exists — cmd_register_current at :2130) when taking a GPU round; (b) Step 0 additionally scans the parent's recent events for a fresh `stage-dispatch` / `epm:run-launched` by another writer and treats it as live-driver evidence. Pair with the sibling filing "spawn_session.py unregister subcommand" for the yield path.

## Constraints

- Workflow-surface only; must not re-introduce the #845 stale-registration wedge (the watcher's stale-registration pass remains the unlock); recursion guard applies.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: dd7eb89d1131
