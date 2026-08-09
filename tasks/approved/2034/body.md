---
title: 'daily-fix: stand-down confirmation + ownership release'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6f8c4b1860f1
- daily-auto-filed
created_at: '2026-08-03T07:00:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): Three same-day coordination
  races: (a) session 472284ce stood down its ladder agent (SendMessage + PID kill)
  and wrote a durable ''STOOD DOWN'' marker -- the agent had already entered production,
  COMPLETED, and committed (aa7d5c867a); the durable record was wrong for ~1.7h until
  a RECORD CORRECTION marker. (b) session a0400dd4''s successor nearly double-dispatched
  duty B while the ''closing'' prior ter'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miners 8/6/4, sessions 472284ce/a0400dd4/75f66748, tasks #1902/#1345).

## Goal

Durable coordination records (stand-downs, handoffs) reflect confirmed state, and no successor dispatches over a live owner.

## Workflow gap

- **Bug observed:** Three same-day coordination races: (a) session 472284ce stood down its ladder agent (SendMessage + PID kill) and wrote a durable 'STOOD DOWN' marker -- the agent had already entered production, COMPLETED, and committed (aa7d5c867a); the durable record was wrong for ~1.7h until a RECORD CORRECTION marker. (b) session a0400dd4's successor nearly double-dispatched duty B while the 'closing' prior terminal was still live and owning the round -- stopped only at the pre-dispatch queue check. (c) session 75f66748 spawned a ladder runner on #1902 while a live owner session already ran it; the runner's own probe caught it.
- **Why it is a workflow gap:** The clause covers channels and durable-state probes for REDOING work, but not confirmation of a stand-down's effect before recording it, nor an explicit release protocol for session-to-session handoffs.
- **Confidence (emitter):** medium (incidents probed by miners: marker sequences + git show of the completed commit)
- verified-at-filing: `grep -n 'stand-down' CLAUDE.md` -> 1 section (clause (c): channel discipline only); 0 hits for 'ownership release'/'release marker'; the (b)/(c) probes cover re-doing a teammate's WORK, not the stood-down-but-completing race.

## Proposed change (refine in planning)

extend the teammate-coordination clause: (i) a stand-down is not effective until CONFIRMED -- probe the agent's state/durable outputs after the SendMessage before writing an aborted/stood-down claim into a durable marker (SendMessage does not preempt an in-flight turn); (ii) session-to-session continuation handoffs post an explicit ownership-RELEASE marker before the successor dispatches; (iii) the ownership probe runs in the ORCHESTRATOR before spawning a runner on a shared artifact path, not only inside the spawned agent.

## Scope / surfaces

- Primary target: `CLAUDE.md`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 6f8c4b1860f1

- workflow_fix_target: CLAUDE.md

