---
title: 'daily-fix: /daily sweep checks retractions before filing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0dea81dbc8e9
- daily-auto-filed
created_at: '2026-07-08T06:59:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): #1101 (8943233a): filed
  by the 07-06 /daily sweep from a marker whose premise the originating #1074 session
  RETRACTED 37 seconds after the sweep mined it; the spawned session archived won''t-fix
  after a 6-minute clarifier pass.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

before filing a route-2 task from a mined marker, the /daily sweep re-reads the source task's newer markers for a retraction/supersession of the mined premise and skips with a note when found

## Workflow gap

- **Bug observed:** #1101 (8943233a): filed by the 07-06 /daily sweep from a marker whose premise the originating #1074 session RETRACTED 37 seconds after the sweep mined it; the spawned session archived won't-fix after a 6-minute clarifier pass.
- **Why it is a workflow gap:** The sweep mines a point-in-time snapshot; without a freshness re-check it files already-dead premises.

## Proposed change

Add to .claude/skills/daily/SKILL.md (route-2 filing steps): before filing from a mined marker/candidate, re-read the source task events.jsonl for markers NEWER than the mined one that retract or supersede the premise; on a hit, skip the filing and record "skipped: premise retracted by <marker> at <ts>" in the daily file. This routes the recursion-guard-parked #1101 candidate.

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: 8943233a (#1101); parked candidate in tasks/archived/1101/events.jsonl.
