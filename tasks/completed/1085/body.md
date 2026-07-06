---
title: 'daily-fix: Step 10d additive-checkout guard-block recovery r'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ad14dbcd813f
- daily-auto-filed
created_at: '2026-07-06T06:59:00Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-05 problem sweep (route 2): Two sessions improvised
  unqualified variants of the Step 10d additive-checkout/restore compound and got
  guard-blocked: #813 ran ''xargs -a ... git checkout issue-813 --'' without -C "$REPO_ROOT"
  - the block also skipped the earlier clause that WRITES /tmp/issue-813-additive-files.txt,
  so the retry failed exit 128 + ''cat: No such file'' before scratch-worktree recovery
  (session d36ef80b, 14:28Z); #105'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-05 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Add to the Step 10d additive-checkout recipe: (a) the guard blocks any unqualified / git-restore variant - use the -C-qualified fence line verbatim; (b) on a guard block, the WHOLE compound was skipped, so regenerate the list file too before retrying (the block message's retry advice omits this).

## Workflow gap

- **Bug observed:** Two sessions improvised unqualified variants of the Step 10d additive-checkout/restore compound and got guard-blocked: #813 ran 'xargs -a ... git checkout issue-813 --' without -C "$REPO_ROOT" - the block also skipped the earlier clause that WRITES /tmp/issue-813-additive-files.txt, so the retry failed exit 128 + 'cat: No such file' before scratch-worktree recovery (session d36ef80b, 14:28Z); #1056 hit the same guard on a git-restore compound (session cc6ec8b5, 18:37Z).
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Doc-level recovery contract; complements #1047's hardening bundle.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- source: /daily 2026-07-05 problem sweep (transcript-mined)
