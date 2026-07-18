---
title: 'workflow-fix: gotcha — smoke-gate floors from realized slice arithmetic'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2382f847309f
created_at: '2026-07-18T14:08:19Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1489 crash-fix r5 (see
  body Provenance block)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1489 (emitting agent: experiment-implementer, crash-fix round 5).

## Goal

Add a gotcha: smoke-gate artifact floors (checkpoint/step counts) must be derived from the realized smoke slice arithmetic, not an assumed row cap.

## Workflow gap

- **Bug observed:** smoke distill JSONL capped at 30 rows realized 2 rows; epochs=1 under 16-row effective batch gave 1 optimizer step against a >=2-checkpoint gate, killing a GCE relaunch at the post-smoke assert (#1489)
- **Why it is a workflow gap:** the trap generalizes to every driver whose post-smoke gate asserts a training-artifact floor — the smoke dial (epochs/steps) must be computed from realized n_rows so the floor holds for ANY realized yield; nothing in `.claude/rules/gotchas.md` warns about calibrating a smoke gate on an assumed slice size.
- **Confidence (emitter):** high
- verified-at-filing: `grep -cin "realized smoke slice\|realized slice" .claude/rules/gotchas.md` → 0 hits in the single named target (absence-of-guard claim; 0-hit IS the evidence) + `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md` → 8 commits, none covering smoke-gate slice arithmetic (2026-07-18)

## Proposed change (candidate diff sketch — refine in planning)

```
+ **Smoke-gate floors must be computed from the REALIZED smoke slice, never an
+ assumed row cap.** A post-smoke assert like ">=2 checkpoints" (the multi-LoRA
+ floor) dies when the realized smoke yield is smaller than the cap the dial was
+ calibrated on (#1489: capped 30, realized 2; epochs=1 @ effective batch 16 ->
+ 1 optimizer step -> 1 checkpoint). Derive the smoke dial from realized
+ arithmetic (epochs = max(1, ceil(K / steps_per_epoch)), K = the gate's floor),
+ keep production geometry untouched, and pin with a fails-pre-fix test + an
+ on-disk checkpoint assert (config math alone does not prove Trainer saves).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'realized.*slice\|smoke.*floor' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan. Mirror memory already exists at `.claude/agent-memory/experiment-implementer/feedback_smoke_gate_realized_slice_arithmetic.md` (commit 235b7890ab) — the gotcha entry is the cross-agent surface.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; gotchas.md row-cap/LESSONS-index consistency maintained (`--check-lessons-index`).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 2382f847309f

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: smoke distill / post-smoke schema assert (scripts/issue1489_dispatch.sh + issue1489_gpu_phase.py _distill_train_cfg)
lesson: Smoke-gate expectations (checkpoint/step counts) must be COMPUTED from the realized smoke slice size, never assumed from a cap comment — the smoke JSONL was capped at 30 rows but the realized gen yield was 2, so epochs=1 under the production 16-row effective batch gave 1 optimizer step against a >=2-checkpoints gate. Derive the smoke dial from realized arithmetic (epochs = max(1, ceil(2/steps_per_epoch))) so the floor holds for ANY realized n, keeping production geometry untouched.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
