---
title: 'daily-fix: poll ticks compute rate/ETA on long same-phase ta'
kind: infra
tags:
- wf-fix
- wf-fix-fp:373b7bbe3701
- daily-auto-filed
created_at: '2026-07-30T07:08:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): #1482''s E2 per-file upload
  ran ~5.4h at GPU idle across five consecutive 30-min poll ticks that each reported
  ''Healthy — E2 upload at shardNN'' without ever computing a rate/ETA; the ~33h projection
  surfaced only when the session finally sized it'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner D-P3 (session ff4119b7, #1482)).

## Goal

Long tail phases must get a throughput read, not just a phase-name liveness read — idle-GPU billing hides behind 'healthy' otherwise.

## Workflow gap

- **Bug observed:** ~5.4h of idle-A100 billing before the ~98 files/h (~33h projection) pathology was sized; recovery (bulk upload_folder commit) then took ~1h. The uploader source fix is already filed as #1824 — this filing is the poll-side detection rule only.
- **Why it is a workflow gap:** poll guidance asks for phase liveness; nothing requires a rate/ETA after repeated same-phase ticks, so a pathological tail reads healthy indefinitely.
- **Confidence (emitter):** medium
- verified-at-filing: miner probe: #1824 exists at proposed (title 'upload_dir_sharded: batch small-file stores into one upload_folder commit...') — source fix routed, do not re-file it (task.py view, 2026-07-30).

## Proposed change (refine in planning)

Add the N-same-phase-ticks rate-read duty to the bg-Bash poll-loop guidance (or pod-side-reporting.md), with the #1482 numbers as the worked example.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 373b7bbe3701
