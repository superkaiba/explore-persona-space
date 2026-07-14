---
title: 'workflow-fix: mechanical ensemble reviewer no-show predicate'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e0e1fca42e1c
- daily-auto-filed
created_at: '2026-07-09T06:56:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The reviewer no-show decision
  at doubled review sites rests on a prose rule only; the orchestrator has no mechanical
  predicate verifying both ensemble verdict markers are present for the round before
  deciding.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #856 (park_form: recursion-guard).

## Goal

Add a mechanical helper predicate (e.g. task_workflow.ensemble_verdicts_present(N, kinds, round) or a task.py subcommand) the orchestrator must run before any reviewer no-show decision.

## Workflow gap

- **Bug observed:** The reviewer no-show decision at doubled review sites rests on a prose rule only; the orchestrator has no mechanical predicate verifying both ensemble verdict markers are present for the round before deciding.
- **Why it is a workflow gap:** Prose-only gates get skipped under context pressure; the precedent (stage_dispatch_should_skip) shows a tested library predicate is strictly stronger and testable.
- **Confidence (emitter):** medium (Phase 2 Alternatives Claude + Codex statistics S2, #856)

## Proposed change (candidate diff sketch — refine in planning)

def ensemble_verdicts_present(issue, kinds, round_n) -> dict[str, bool]: read events.jsonl, return per-kind presence for the round; SKILL.md no-show step cites it as the required pre-decision check.

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py, .claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py, .claude/skills/issue/SKILL.md
- origin: parked candidate on task #856 at 2026-07-02T16:29:49Z

parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target recursion guard — see .claude/rules/workflow-fix-on-bug.md § Recursion guard. Candidate surfaced by the Phase 2 critic ensemble (Alternatives Claude + Codex statistics S2 concern): a MECHANICAL helper predicate (e.g. task_workflow.ensemble_verdicts_present(N, kinds, round) or a task.py subcommand) the orchestrator must run before any reviewer no-show decision would be strictly stronger than the prose rule this task ships (precedent: stage_dispatch_should_skip). target_file: src/explore_persona_space/task_workflow.py, .claude/skills/issue/SKILL.md. confidence: medium. NOT routed by this session (recursion guard); logged for a later orchestrator/human pass.
