---
title: 'daily-fix: crash-fix relaunch asserts fix commit on pod + wi'
kind: infra
tags:
- wf-fix
- wf-fix-fp:22f16a77ea22
- daily-auto-filed
created_at: '2026-07-06T06:58:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-05 problem sweep (route 2): #779 stale-code launch:
  launch #1 ran the commit BEFORE the implementer''s val-lambda fix, hit the known
  GCV ridge degeneracy at n~d (val R^2 -4.7 vs ~0.6 expected) and checkpointed the
  garbage; the restart faithfully resumed the poisoned checkpoints. Recovery required
  wiping checkpoints and relaunching at current HEAD with a 75s health probe reporting
  pod git HEAD + first val-selection lines. Sess'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-05 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Extend the crash-fix-rounds fix-engaged-signal contract: after a code-fix round, the relaunch (a) asserts the specific fix commit is present on the pod (git HEAD / merge-base probe) before dispatch, and (b) wipes checkpoints written by the stale-code run so a resume cannot inherit poisoned state; name both in the declared fix-engaged signal.

## Workflow gap

- **Bug observed:** #779 stale-code launch: launch #1 ran the commit BEFORE the implementer's val-lambda fix, hit the known GCV ridge degeneracy at n~d (val R^2 -4.7 vs ~0.6 expected) and checkpointed the garbage; the restart faithfully resumed the poisoned checkpoints. Recovery required wiping checkpoints and relaunching at current HEAD with a 75s health probe reporting pod git HEAD + first val-selection lines. Session 5664c4f8, task #779, 2026-07-06.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/rules/crash-fix-rounds.md`
- The experimenter pre-launch sync mandate exists; this adds the assert-the-fix-commit + checkpoint-hygiene specifics for crash-fix rounds.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: .claude/rules/crash-fix-rounds.md
- source: /daily 2026-07-05 problem sweep (transcript-mined)
