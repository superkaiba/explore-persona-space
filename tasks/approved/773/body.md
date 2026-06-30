---
title: 'workflow-fix: protect active-task input caches from vm_disk_guard reclaim'
kind: infra
tags:
- wf-fix
- wf-fix-fp:45d6ec35c1c1
created_at: '2026-06-30T21:51:05Z'
has_clean_result: false
origin_prompt: what is the infra friction and how can we avoid it in the future -
  send a subagent to investigate
---
## Overview / Motivation
Auto-filed from the predictor-line (#742/#761/#763) infra-friction retrospective (subagent, 2026-06-30). Related in-flight: #770 (RunPod no-port wedge billing), #681 (data-disk bind cutover), #667 (GCP frozen-phase), #769 (CUDA-frag OOM).

## Goal
Never reap a terminal task's hf_dl/store cache if a currently-ACTIVE task declares it as a planned input (artifact-reuse provenance is readable); at minimum never reap data/issue_<M>/store/ referenced by another active issue's plan.

## Workflow gap
- Bug observed: vm_disk_guard.py --apply reaped #742's #658 UltraChat input artifacts during a panic clean (treated as reclaimable terminal-task cache), causing #742's third launch to die on FileNotFoundError.

## Scope / surfaces
- Target: scripts/clean_experiment_downloads.py, CLAUDE.md

## Constraints / invariants
- Workflow-surface only; ruff + workflow_lint pass; add a test where practical.

## Provenance
- workflow_fix_target: scripts/clean_experiment_downloads.py, CLAUDE.md
- fingerprint: 45d6ec35c1c1

Raised by the predictor-line infra retrospective (2026-06-30).
