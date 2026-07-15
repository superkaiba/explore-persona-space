---
title: 'daily-fix: honor user inline override; commit same turn'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7de5801899d3
- daily-auto-filed
created_at: '2026-07-15T06:51:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): follow-up routing was overridden
  to inline by the user 5+ times in one day (4 spawned /issue sessions killed within
  ~2 min: #658, #1092, #1090, #952) — the user-chat inline carve-out has no affordance
  for an explicit user inline ask on a GPU-needing follow-up, ''as followups'' phrasing
  pulls 0-GPU asks to the spawn path, and the 07-06 #779 inline round left its eval
  JSON uncommitted + report stale f'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep — the day's single most-corrected pattern. Evidence: (i) #658 + #1092 sessions spawned 21:19Z, stopped ~2 min later, "can you just run them inline"; (ii) #825: "are you running it inline?" then "just run it inline" on a 2-3 GPU-h follow-up (21:18Z); (iii) #952: session spawned 04:44Z, stopped 04:46Z, "run it inline so it is faster. run in parallel as much as possible". Separately the #779 07-06 inline round posted results but never committed `fair_comparison.json` (8 days modified-but-uncommitted) and never updated the report — Thomas discovered the stale Result 3 himself (21:19Z).

## Goal

extend CLAUDE.md § User-chat inline free analysis: (a) an explicit user inline/faster ask is honored as an override even when the follow-up needs GPU (post the override epm:progress marker + pod-safety pre-launch signals); (b) a user-chat 0-GPU-h ask on existing artifacts defaults inline even when phrased 'as followups'; (c) inline-round completion requires committing the round's eval JSONs + folding the result in the same turn

## Workflow gap

- **Bug observed:** follow-up routing was overridden to inline by the user 5+ times in one day (4 spawned /issue sessions killed within ~2 min: #658, #1092, #1090, #952) — the user-chat inline carve-out has no affordance for an explicit user inline ask on a GPU-needing follow-up, 'as followups' phrasing pulls 0-GPU asks to the spawn path, and the 07-06 #779 inline round left its eval JSON uncommitted + report stale for 8 days
- **Why it is a workflow gap:** the carve-out is scoped to 0-GPU-on-existing-artifacts with no user-override affordance, so rule-following behavior contradicts the user's expressed preference during interactive iteration; and the carve-out's completion contract does not require committing artifacts, so inline rounds can leave stale reports.
- **Confidence:** high (5+ corrections, 4 killed sessions, 1 stale-report incident)
- verified-at-filing: `grep -n "run it inline\|inline override" CLAUDE.md` -> 0 hits (absence of the override affordance); the carve-out text at § Routing "User-chat inline free analysis" prescribes routing to the follow-up loop for any needs-gpu ask (2026-07-15).

## Proposed change

Three additions to the carve-out (and its pod-safety deviation case already covers the pre-launch signals for GPU inline runs): the explicit-override clause, the as-followups-defaults-inline clause for 0-GPU asks, and the same-turn commit clause. Also mirror (a)/(b) in `.claude/agents/research-pm.md` if the planner finds the PM dispatch path is where the spawns originated.

## Constraints

- Workflow-surface only; do not weaken pod-safety signal duties for GPU inline runs; recursion guard applies.

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: 7de5801899d3
