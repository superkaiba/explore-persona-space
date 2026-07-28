---
title: 'daily-fix: inline dispatch arms detached handoff by default'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ddd557608861
- daily-auto-filed
created_at: '2026-07-28T07:04:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): after dispatching a pod-side
  gate in a user-directed inline round, the session''s ack did not volunteer terminal-lifecycle
  info; Thomas had to ask ''can i safely close this terminal?'' then explicitly order
  ''make it so everything continues in the background'' before the pod-side handoff
  chain was armed'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session ca5c6f64 (#1689 inline R16 round), 2026-07-28T00:54Z (miner I P8; recurrence of the detach-transparency rule — 7 asks across 3 sessions on 07-24 already).

## Goal

User-directed inline rounds should be detached-by-default with the lifecycle stated at dispatch, not after the user asks.

## Workflow gap

- **Bug observed:** the R16 equivalence-gate dispatch ack omitted lifecycle; Thomas asked the can-I-close question (answered well in-turn) and then had to order the detached handoff explicitly — two extra user turns, ~25 min where closing the terminal would have orphaned the gate->relaunch handoff.
- **Why it is a workflow gap:** the SOUL.md detach-transparency rule covers VOLUNTEERING lifecycle info; the project CLAUDE.md inline-override duties do not require ARMING the detached handoff by default at dispatch. unverified hypothesis — verify at plan time: whether an existing CLAUDE.md clause partially covers this (the inline-override duty list is long; re-read it before editing).
- **Confidence (emitter):** medium
- verified-at-filing: recurrence evidence: the 07-24 SOUL.md rule (7 asks / 3 sessions) + this session's two turns (miner-quoted verbatim). Gap location: compose-time grep of CLAUDE.md's inline-override block found pod-safety/compute duties but no detached-handoff-default clause.

## Proposed change (candidate diff sketch — refine in planning)

In project `CLAUDE.md` (§ user-chat inline override duties): add — 'a user-directed inline dispatch of pod/VM-side work ARMS the detached handoff chain by default (pod-side watch -> relaunch/harvest breadcrumbs), and the dispatch ack states the lifecycle line: what runs detached, what dies with this terminal, safe-to-close verdict.'

## Scope / surfaces

- Primary target: `CLAUDE.md` (inline-override duties)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: ddd557608861

- workflow_fix_target: CLAUDE.md
