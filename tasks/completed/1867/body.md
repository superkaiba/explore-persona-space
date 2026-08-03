---
title: 'daily-fix: task.py address-concern accepts --note alias'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8ae042147846
- daily-auto-filed
created_at: '2026-07-30T07:10:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): Two sessions on consecutive
  days (07-28, 07-29) called address-concern --note and got argparse exit 2 — every
  other note-bearing subcommand uses --note; the asymmetry is the trap'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner H-P5 (probed); second same-flag miss in 2 days (CLAUDE.md already documents the 07-28 one; tonight's route-1 doc fix adds the required --by/--round flags)).

## Goal

Make the recurring flag-convention slip fail soft: --note aliases --summary.

## Workflow gap

- **Bug observed:** argparse exit 2 on --note; the flag surface asymmetry (post-marker/set-status take --note; address-concern takes --summary) recurs across sessions.
- **Why it is a workflow gap:** CLI convention consistency is the trap-remover; a doc line alone did not prevent the second miss.
- **Confidence (emitter):** medium
- verified-at-filing: `uv run python scripts/task.py address-concern --help` -> only --summary (required --by/--round confirmed) (2026-07-30, this run).

## Proposed change (refine in planning)

Add the alias (dest=summary); keep --summary canonical in docs.

## Scope / surfaces

- Primary target: `scripts/task.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/task.py
- fingerprint: 8ae042147846
