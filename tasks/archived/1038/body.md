---
title: 'daily-held: schedule #681 bind-migration cutover + disk sign-off'
kind: infra
tags:
- daily-held
created_at: '2026-07-04T23:02:06Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — disk-cutover-signoff (fp 9851cc2eed1c)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

Thomas: (1) pick a quiet window for the #681 cutover (recipe already in CLAUDE.md 'Disk hygiene' — LOCK, move worktrees, bind-mount, fstab, env flags, seed quotas, unlock); (2) approve/deny the itemized safe deletions from the 07-03 diagnosis session.

## Workflow gap

- **Bug observed:** Boot disk / hit 95-96%% on 07-03 (6.4-17 GiB free overnight; 92%% / 40G free as of 07-04); policy-safe cleanup recovers ~0 because the accumulation is real: .claude/worktrees ~97G still on the boot disk (the #681 bind-migration to /mnt/eps-data remains deferred), /tmp/i779_mirror_dl ~59G, ~/.cache/huggingface ~60G. The 07-03 diagnosis session identified ~106G of verified-safe deletions that all need sign-off.
- **Why it matters:** CARVE-OUT: destructive deletions + a live-VM migration window are user decisions (route 3).
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `VM ops (destructive deletions + live-VM migration window)`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 3 — judgment call, needs-human).

## Provenance

- workflow_fix_target: VM ops (destructive deletions + live-VM migration window)
- fingerprint: 9851cc2eed1c
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
