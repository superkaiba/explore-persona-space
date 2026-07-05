---
title: 'workflow-fix: gotchas.md tiny-real CPU e2e before first GPU launch (mock-seam
  smoke gap)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f0907d740426
created_at: '2026-07-04T11:06:50Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #906 r15 pivot (four shape bugs,
  four GPU cycles)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha candidate raised on task #906 (emitting agent: experiment-implementer, r15 pivot round).

## Goal

Add a gotchas.md entry (and consider a smoke-gate rule extension) for the tiny-real CPU e2e pattern: before the FIRST GPU launch of any multi-stage experiment driver, a CPU pass of the FULL production path (from-config small same-arch model, real train/verify/upload bodies, only GPU-scale weights + the remote Hub boundary faked) — mock-seam smokes surface shape bugs one per GPU cycle.

## Workflow gap

- **Bug observed:** #906 burned four ~1.5h GPU cycles on four DISTINCT production shape bugs (Anthropic message shape -> TrainLoraConfig kwarg -> row truncation -> LoraConfig-object-vs-dict), each one pipeline stage deeper; every mock-seam smoke and contract-test round PASSed beforehand.
- **Why it is a workflow gap:** the Step 6d.0/6d.0-bis smoke gates accept seam-stubbed smokes; no rule requires the full path to execute with REAL library types once on CPU before a GPU provision. The tiny-real pattern (tests/test_issue906_tiny_real_e2e.py, 118s CPU) is now a worked example.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md: "Mock-seam smokes surface shape bugs one per GPU cycle (#906 r11-r15: four distinct bugs, four pod cycles). Before the first GPU launch of a multi-stage driver, run a tiny-real CPU e2e of the FULL production path: real tokenizer, real train engine + callbacks, real adapter round-trip, real verify/upload bodies; fake ONLY GPU-scale weights (from-config 2-layer same-arch model, real vocab-id space) and the remote Hub boundary (signature-bound). Traps: assert_gauge_free_adapter_config takes the parsed adapter_config.json DICT (never a PEFT LoraConfig object); TrainingArguments(bf16=True) hard-raises on CPU — use the TrainLoraConfig.bf16 knob."
+ Consider: extend the Step 6d.0-bis end-to-end smoke gate wording (issue SKILL.md) to name the tiny-real standard for kind:experiment/batch drivers with faked seams (architectural judgment for the spawned session's planner).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Secondary (planner's call): .claude/skills/issue/SKILL.md Step 6d.0-bis wording; .claude/agents/experiment-implementer.md smoke section.

## Constraints / invariants

- Workflow-surface only; scripts/workflow_lint.py passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 and carries a workflow_fix_target: Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: f0907d740426

(verbatim lesson: see epm:failure-lesson v5 on task #906 — mock-seam smokes discover shape bugs one per GPU cycle; tiny-real CPU pass of the full path is the terminal fix.)
