---
title: 'daily-fix: set-body warns when experiment loses Goal H2'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c5c8cb7b7ad1
- daily-auto-filed
created_at: '2026-07-08T06:59:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): #1112 (events.jsonl 17:04:58Z):
  the re-scope set-body dropped the ## Goal H2 from a kind:experiment body ("Mechanical
  repair: re-ran set-goal ... to restore the missing ## Goal H2 the re-scope set-body
  dropped"); the Goal gate caught it later at the cost of a bounce + repair.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

task.py set-body emits a loud warning (planner may prefer refuse + an override flag) when a kind:experiment body update removes the ## Goal H2 present in the prior body

## Workflow gap

- **Bug observed:** #1112 (events.jsonl 17:04:58Z): the re-scope set-body dropped the ## Goal H2 from a kind:experiment body ("Mechanical repair: re-ran set-goal ... to restore the missing ## Goal H2 the re-scope set-body dropped"); the Goal gate caught it later at the cost of a bounce + repair.
- **Why it is a workflow gap:** The Goal is the canonical target every downstream subagent reads; silent loss costs a later gate bounce + repair.

## Proposed change

In task_workflow set-body: when the task is kind: experiment and the PRIOR body contains a ## Goal H2 but the new body does not, print a loud WARNING (or refuse + --allow-goal-drop; planner decides). Small guard + test.

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: tasks/*/1112/events.jsonl 17:04:58Z.
