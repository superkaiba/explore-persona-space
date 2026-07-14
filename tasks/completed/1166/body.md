---
title: 'workflow-fix: planner.md: prescribe verdict-lattice declarat'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cecf508f3270
- daily-auto-filed
created_at: '2026-07-09T06:58:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): c20-class tier-1 review
  gates (FAIL-capable on both defect classes) require the ''DISJOINT and exhaustive:
  <label> ⇔ <predicate>; ...'' verdict-lattice declaration form, but no planner-side
  spec tells planners to write it — only 2 corpus files carry it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1006 by a recursion-guarded workflow-fix session.

## Goal

Add a short verdict-lattice subsection to planner.md (§3 Hypothesis / verdict-lattice guidance; possibly .claude/rules/planner-section-reference.md) prescribing the ⇔ declaration form plus the 'N/A — no registered verdict lattice' escape.

## Workflow gap

- **Bug observed:** c20-class tier-1 review gates (FAIL-capable on both defect classes) require the 'DISJOINT and exhaustive: <label> ⇔ <predicate>; ...' verdict-lattice declaration form, but no planner-side spec tells planners to write it — only 2 corpus files carry it.
- **Why it is a workflow gap:** Plans only adopt the registered-lattice form when a critic bounces them; a planner-side spec makes adoption proactive.
- **Confidence (emitter):** medium
- **Sweep verification (2026-07-08):** grep for 'verdict.lattice', '⇔', and 'DISJOINT and exhaustive' in .claude/agents/planner.md and .claude/rules/planner-section-reference.md returns nothing on main (2026-07-08); no open proposed/on_hold task mentions a verdict lattice.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; add a ~10-line subsection to planner.md §3 naming the '<label> ⇔ <predicate>' form, the disjoint+exhaustive requirement, and the N/A escape)

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: .claude/agents/planner.md
- origin: parked candidate on task #1006 at 2026-07-04T13:33:32Z

Verbatim parked note:

> parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). Candidate (from plan v3 §15, surfaced by the planner): document the ⇔ registered-lattice form in the planner spec so future plans adopt it proactively. target_file: .claude/agents/planner.md (§3 Hypothesis / verdict-lattice guidance; possibly .claude/rules/planner-section-reference.md). bug_observed: c20's tier-1 (FAIL-capable both defect classes) requires the 'DISJOINT and exhaustive: <label> ⇔ <predicate>; ...' form, but no planner-side spec tells planners to write it — only 2 corpus files carry it today. proposed_change: add a short verdict-lattice subsection to planner.md prescribing the ⇔ declaration form + the 'N/A — no registered verdict lattice' escape. confidence: medium. related_task: #1006. Routing: parked for the next non-workflow-fix orchestrator/human pass — NOT auto-filed from this session.
