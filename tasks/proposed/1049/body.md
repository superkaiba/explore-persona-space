---
title: 'daily-fix: document pod-to-HF upload wedge escalation ladder'
kind: infra
tags:
- wf-fix
- wf-fix-fp:161ae961eea5
- daily-auto-filed
created_at: '2026-07-05T07:02:37Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): #931 spent ~50 min on 2026-07-04
  breaking three successive pod-side HF upload wedges (xet stall -> hf_transfer stall
  -> CDN-route-specific dead end) before landing the working recovery: rsync 9.8 GB
  to the VM and upload from there. The #515 xet-CDN workaround note covers only the
  first rung.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Document the full three-rung wedge escalation ladder (HF_XET_DISABLE=1 -> HF_HUB_ENABLE_HF_TRANSFER=0 -> rsync-to-VM reroute) in upload-policy.md next to the #515 note so the next wedged session runs the ladder instead of rediscovering it.

## Workflow gap

- **Bug observed:** #931 spent ~50 min on 2026-07-04 breaking three successive pod-side HF upload wedges (xet stall -> hf_transfer stall -> CDN-route-specific dead end) before landing the working recovery: rsync 9.8 GB to the VM and upload from there. The #515 xet-CDN workaround note covers only the first rung.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/rules/upload-policy.md`
- Session: 2f37cc81 (#931, ~06:00-06:52 UTC).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/rules/upload-policy.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
