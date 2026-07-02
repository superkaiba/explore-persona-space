---
title: 'workflow-fix: gotchas entry — TrainerCallback subclass + real-trainer-path
  smoke'
kind: infra
tags:
- wf-fix
created_at: '2026-07-02T02:11:02Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #816 crash-fix r3: hand-rolled
  HF callback must subclass TrainerCallback; GPU-bound-phase smoke must traverse real
  SFTTrainer.__init__'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson `gotcha_candidate: yes` on task #816 (emitting agent: experiment-implementer, crash-fix round 3).

## Goal

Add a gotchas entry: hand-rolled HF Trainer callbacks must subclass `transformers.TrainerCallback`, and GPU-bound-phase smokes must traverse the REAL `SFTTrainer.__init__` lifecycle (tiny same-arch model on CPU), not dry-run/import-check substitutes.

## Workflow gap

- **Bug observed:** #816's production run crashed at `SFTTrainer.__init__` (`AttributeError: 'PreventativeSteeringCallback' object has no attribute 'on_init_end'`) after 53/53 steering cells had already burned GPU-hours — the callback's docstring claimed TrainerCallback but the class had no base; the round-1 CPU smoke substitutes (dry-run/import-check for the GPU-bound phase) never construct a real trainer, so the lifecycle-contract bug was invisible until production.
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` documents codebase traps for training/eval code; this trap (a) recurs for ANY custom callback and (b) exposes a general smoke blind spot for GPU-bound phases.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/rules/gotchas.md:
+ ## Hand-rolled Trainer callbacks + GPU-bound-phase smokes
+ - A callback passed to any HF Trainer MUST subclass transformers.TrainerCallback
+   (CallbackHandler.call_event fires on_init_end inside Trainer.__init__; only the
+   subclass inherits no-op defaults). Incident #816.
+ - A GPU-bound phase's CPU smoke substitute (dry-run/import-check) never traverses
+   SFTTrainer.__init__ — smoke the REAL trainer path on a tiny same-arch model on
+   CPU (max_steps=1-2) instead.
Also consider: experiment-implementer.md smoke checklist line naming the real-trainer-path requirement when callbacks= appears.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Secondary (planner's call): `.claude/agents/experiment-implementer.md` smoke checklist.

## Constraints / invariants

- Workflow-surface only. Lint passes.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: (computed at filing)
