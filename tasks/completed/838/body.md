---
title: 'workflow-fix: planner/critic subagent spawns still autocompact-thrash post-#829'
kind: infra
tags:
- wf-fix
- wf-fix-fp:23e7e522d498
created_at: '2026-07-02T06:35:51Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by /issue 834 orchestrator: planner/critic-type
  subagents thrash at spawn; see body'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #834 (emitting agent: /issue orchestrator).

## Goal

Diagnose the spawn-time context composition for the `planner` / `critic` agent
types and shed enough always-on weight that a fresh spawn can read a ~450-line
plan plus a ~650-line module without autocompact thrashing.

## Workflow gap

- **Bug observed:** planner- and critic-type subagent spawns autocompact-thrash
  ("context refilled to the limit within 3 turns of the previous compact, 3
  times in a row") within 4-9 tool uses, even after the #829 spec trim
  (planner.md 95,759→63,888 B). 5 of 6 such spawns failed while driving task
  #834 on 2026-07-02: 2× planner (plan drafting), 1× planner-as-fact-checker,
  2× critic (statistics + alternatives lenses). The SAME briefs re-run as
  `general-purpose` agents (16 KB-class spec) succeeded at ~160-170K total
  tokens each, so the work itself fits — the agent-type baseline does not.
- **Why it is a workflow gap:** the per-agent-type context composition (spec +
  memory frontmatter + always-on CLAUDE.md/import pile + whatever else the
  harness pins at spawn) leaves too little compactable headroom for the
  role's own mandated reads; #829's trim was insufficient for these two roles.
  Every /issue pipeline stage that spawns them (adversarial-planner Phases 1,
  1.5, 2) silently degrades to crashed rounds + orchestrator fallbacks.
- **Confidence (emitter):** medium (symptom high-confidence + reproducible;
  root-cause attribution between spec size, memory loading, and harness pin
  composition not yet isolated).

## Proposed change (candidate diff sketch — refine in planning)

- Measure actual spawn-time baseline tokens for `planner` / `critic` /
  `general-purpose` (e.g. a probe agent that reports its own context usage on
  turn 1) to isolate the dominant term.
- Trim `.claude/agents/planner.md` (63.9 KB) and `.claude/agents/critic.md`
  (65.5 KB) further: move worked examples / incident post-mortems / per-lens
  long-form guidance into on-demand reference files loaded by pointer, keeping
  the always-on role spec ≤ ~20 KB (the empirically-working general-purpose
  class).
- Tighten the #829 size lint threshold to whatever the probe shows is safe.

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`, `.claude/agents/critic.md`
- Secondary: `scripts/workflow_lint.py` (size-lint threshold from #829);
  consider the same treatment for other >60 KB specs (analyzer.md 112 KB,
  clean-result-critic.md 104 KB, code-reviewer.md 69 KB) if the probe confirms
  spec size is the dominant term.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- No role-content loss: moved material stays reachable via explicit pointers.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/planner.md, .claude/agents/critic.md
- fingerprint: 23e7e522d498

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/planner.md, .claude/agents/critic.md
bug_observed: planner- and critic-type subagent spawns autocompact-thrash within 4-9 tool uses even after the #829 spec trim (5 of 6 spawns failed on task #834, 2026-07-02)
why_workflow_gap: the agent-type spawn-time context composition leaves too little compactable headroom for the role's own mandated reads; #829's trim was insufficient for these two roles
proposed_change: diagnose spawn-time context composition for planner/critic agent types and shed enough always-on weight that a fresh spawn can read a 450-line plan plus a 650-line module without autocompact thrashing
diff_sketch: |
  - probe spawn-time baseline tokens per agent type
  - move worked examples / incident post-mortems out of planner.md + critic.md
    into on-demand reference files (target ≤ ~20 KB always-on)
  - tighten the #829 workflow_lint size threshold accordingly
confidence: medium
related_task: #834
<!-- /workflow-fix-candidate -->
