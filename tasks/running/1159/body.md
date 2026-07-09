---
title: 'workflow-fix: Split clean-result-critic.md; add pointer-cove'
kind: infra
tags:
- wf-fix
- wf-fix-fp:07216fd7f2d6
- daily-auto-filed
created_at: '2026-07-09T06:57:43Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): (1) clean-result-critic.md
  is a 108,106-byte monolith (grandfathered at 108,900; the lint table itself names
  ''SPEC.md-dedup trim'' as the standing follow-up), and codex-clean-result-critic.md
  inlines the full spec content — the same context-thrash class #838/#850 fixed for
  analyzer.md/critic.md. (2) No workflow_lint check verifies per-section pointer coverage
  for <agent>-section-reference.md rule f'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #850 by a recursion-guarded workflow-fix session.

## Goal

(1) Split clean-result-critic.md into a slim spec + pointer-loaded lens-reference (the #850 analyzer.md pattern, commit 6b1b57472b), including the codex-twin inline-composition contract redesign; (2) add a workflow_lint check that every real non-fenced H2 in a <agent>-section-reference.md file has a pointer line in its owning agent spec naming the exact heading (fence-aware).

## Workflow gap

- **Bug observed:** (1) clean-result-critic.md is a 108,106-byte monolith (grandfathered at 108,900; the lint table itself names 'SPEC.md-dedup trim' as the standing follow-up), and codex-clean-result-critic.md inlines the full spec content — the same context-thrash class #838/#850 fixed for analyzer.md/critic.md. (2) No workflow_lint check verifies per-section pointer coverage for <agent>-section-reference.md rule files, the class that produced the #850 r1 Codex MAJOR (Step 3.5 relocated but not pointer-reachable).
- **Why it is a workflow gap:** Oversized agent specs autocompact-thrash the reviewers that load them, and relocated-but-unreachable reference sections silently drop review coverage; both classes recurred on #850 itself.
- **Confidence (emitter):** medium
- **Sweep verification (2026-07-08):** Verified 2026-07-08: clean-result-critic.md is 108,106 B and still monolithic (the #850 split touched analyzer.md only, commit 6b1b57472b); no per-section pointer-coverage check exists in workflow_lint.py (greps for 'section-reference'/'pointer' hit only the vm-thread-cap floors and analyzer-section-reference count entries). No open task covers either. Planner note: workflow v2 retires clean-result-critic + twin AT v1 DRAIN (CLAUDE.md § Workflow v2) — v1 is still the default and every current task loads this file, but the planner should weigh split effort against the retirement horizon and may legitimately deflect leg (1) while keeping leg (2).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; leg 1 follows the analyzer.md split shape: slim spec + .claude/rules/clean-result-critic-lens-reference.md, codex composer loads per-lens spans; leg 2: new check_section_reference_pointer_coverage over .claude/rules/*-section-reference.md, fence-aware H2 enumeration, owning-agent pointer grep)

## Scope / surfaces

- Primary target: `.claude/agents/clean-result-critic.md, .claude/agents/codex-clean-result-critic.md, scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: .claude/agents/clean-result-critic.md, .claude/agents/codex-clean-result-critic.md, scripts/workflow_lint.py
- origin: parked candidate on task #850 at 2026-07-02T16:41:51Z

Verbatim parked note:

> parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — this session is itself a workflow-fix session; candidates LOGGED + notified, not auto-routed). TWO candidates for the next orchestrator/human pass: (1) [from plan §6, planner prose follow-up] split clean-result-critic.md (104,113 B, grandfathered at 107_000) into slim spec + lens-reference, INCLUDING the codex-twin inline-composition contract redesign (codex-clean-result-critic.md inlines the full spec content at ~L246/276/476) — sibling of #838/#850, same thrash class. (2) [from code-review r2, mechanizable observation] add a workflow_lint check: per-section pointer coverage for <agent>-section-reference.md rule files (every real non-fenced H2 in a reference file must have a pointer line in its owning agent spec naming the exact heading) — the r1 Codex MAJOR (Step 3.5 relocated but not pointer-reachable) is exactly the class this would catch mechanically; fence-aware to skip code-fenced pseudo-headings.
