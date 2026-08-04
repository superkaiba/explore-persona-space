---
title: 'daily-fix: /daily filed bodies cite marker per mechanism cla'
kind: infra
tags:
- wf-fix
- wf-fix-fp:25fa51dc0c5b
- daily-auto-filed
created_at: '2026-07-30T07:12:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): Two 07-28-filed bodies
  carried unverified mechanism claims refuted in review (#1814''s ''12 rows killing
  the engine'' conflation; #1798''s unverified absence claim) — each burned a critic
  REVISE round; the #1677 labeling duty exists but did not bind these compositions'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miners C-P1 (#1814) and E-P9 (#1798) — two same-day recurrences of the #1677 class).

## Goal

Filed bodies are premises for spawned planners; a mechanism claim without a marker citation or the label reliably burns a critic round.

## Workflow gap

- **Bug observed:** #1814's body (route-2 driver, 07-28) asserted a single-mechanism story #1738's events do not support (critic REVISE r1, split in v2); #1798's plan v1 inherited an unverified absence claim (critic REVISE r1, dropped in v2).
- **Why it is a workflow gap:** the #1677 labeling duty is stated but not mechanically anchored to the miner schema's probed/inferred field — the translation step is where both failures happened.
- **Confidence (emitter):** medium
- verified-at-filing: miner probe (C-P1): `grep -n -i 'kresample|12 rows|engine' tasks/running/1814/body.md` -> L28 carries the conflated claim verbatim (composed by the filer, not the planner).

## Proposed change (refine in planning)

Add to the route-2/3 compose step: a mechanism claim from a miner finding marked 'inferred — not probed' MUST ship as `unverified hypothesis — verify at plan time: ...`; a probed finding cites the probe command; incident claims cite the marker ts+kind where one exists.

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/daily/SKILL.md
- fingerprint: 25fa51dc0c5b
