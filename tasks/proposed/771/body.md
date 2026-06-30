---
title: 'workflow-fix: warn when #681 data-disk attached but worktree bind inert'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5e922dbbc041
created_at: '2026-06-30T21:50:32Z'
has_clean_result: false
origin_prompt: what is the infra friction and how can we avoid it in the future -
  send a subagent to investigate
---
## Overview / Motivation
Auto-filed from the predictor-line (#742/#761/#763) infra-friction retrospective (subagent, 2026-06-30). Related in-flight: #770 (RunPod no-port wedge billing), #681 (data-disk bind cutover), #667 (GCP frozen-phase), #769 (CUDA-frag OOM).

## Goal
Add a once-per-process WARN in new_worktree.sh when EPS_VM_DATA_DISK_PATH is a live mount but .claude/worktrees is NOT a bind mountpoint AND EPS_WORKTREE_REQUIRE_BIND!=1 (attached-but-inert), pointing at the cutover; document the one-time #681 cutover (bind + flip the two flags) in CLAUDE.md Disk hygiene. NOTE: the actual bind-migration ops cutover is separate (deferred to a quiet window).

## Workflow gap
- Bug observed: #681 (dedicated data disk + ext4 quotas) is marked completed but on the live VM .claude/worktrees is NOT bind-mounted onto /mnt/eps-data and EPS_WORKTREE_REQUIRE_BIND/EPS_WORKTREE_ASSIGN_QUOTA are unset, so worktree caches still land on the 485G boot disk with no quota — proximate cause of three /=0.0GiB CRITICAL events on 2026-06-30.

## Scope / surfaces
- Target: scripts/new_worktree.sh, CLAUDE.md

## Constraints / invariants
- Workflow-surface only; ruff + workflow_lint pass; add a test where practical.

## Provenance
- workflow_fix_target: scripts/new_worktree.sh, CLAUDE.md
- fingerprint: 5e922dbbc041

Raised by the predictor-line infra retrospective (2026-06-30).
