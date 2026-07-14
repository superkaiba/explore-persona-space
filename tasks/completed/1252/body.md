---
title: 'trigger-dense reviews: return text = verdict + pointer only'
kind: infra
tags:
- wf-fix
- wf-fix-fp:eaf762471cc0
- daily-auto-filed
created_at: '2026-07-10T06:55:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): a reviewer''s PASS summary
  recapping guard findings wedged the ORCHESTRATOR with 3 consecutive usage-policy
  refusals (1152, 09:38-09:49Z)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-09 from the nightly transcript problem sweep (miner 04, P2 — the one orchestrator-killing incident of the day).

## Goal

Tighten `.claude/rules/trigger-dense-review.md` (and the reviewer briefing guidance) so a review-role subagent's RETURN TEXT on a trigger-dense/guard-security task is the verdict + a marker/file pointer ONLY — never a findings recap — because the recap itself wedges the ORCHESTRATOR with usage-policy refusals when it enters the parent context.

## Workflow gap

- **Bug observed:** On #1152 (session a8931290, 2026-07-09T09:38–09:49Z) the re-spawned code-reviewer posted its verdict marker correctly, then returned a PASS summary that RECAPPED the guard findings (hook-bypass shapes) in its final text; the moment that recap entered the orchestrator's context, the orchestrator hit 3 consecutive usage-policy-refused turns and died (the #1074 wedge shape); the watcher respawn (session 2418bba6, 10:13Z) completed the task.
- **Why it is a workflow gap:** `trigger-dense-review.md` (shipped today via #1185) already mandates verdict-marker-first and reference-by-file:line, but does not forbid a findings RECAP in the subagent's final return text — the one channel guaranteed to enter the parent orchestrator's context. The rule protects the reviewer's own turn but not the parent's.

## Proposed change (refine in planning)

Add to `trigger-dense-review.md` (and the code-reviewer/reconciler dispatch briefs in the /issue SKILL where they cite the rule): "final return text on a trigger-dense artifact = verdict word + marker/verdict-file pointer + counts only; NO finding descriptions, NO quoted command shapes, NO bypass-shape summaries — the parent reads the verdict file with windowed reads if needed." Coordinate with the sibling filing `issue-skill-excerpt-file-briefs` (same rule file's orchestrator-side leg).

## Scope / surfaces

- Primary target: `.claude/rules/trigger-dense-review.md`
- Secondary: `.claude/agents/code-reviewer.md`, `.claude/agents/reconciler.md` (one-line pointer)

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: eaf762471cc0

- workflow_fix_target: .claude/rules/trigger-dense-review.md
