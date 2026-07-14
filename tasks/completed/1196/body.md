---
title: 'workflow-fix: corpus WARN on H1 vs frontmatter title drift'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3e12b9f968a7
- daily-auto-filed
created_at: '2026-07-09T07:00:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Retitles after the final
  gate (awaiting_promotion post-critic-PASS or completed bodies, the #432/#458 divergence
  shape) are never re-verified: no verifier revisits those bodies and the corpus auditor
  does not compare H1 vs frontmatter title.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08, slice 5) from a
candidate parked on task #1110 by a recursion-guarded workflow-fix session.

## Goal

Add a corpus-wide WARN-level H1-vs-frontmatter-title sync check to the clean-results body-discipline auditor, so post-gate title drift on parked/grandfathered bodies gains a mechanical surface.

## Workflow gap

- **Bug observed:** The #1110 check binds only at gate-time verify_task_body.py runs; a retitle AFTER the final gate (on awaiting_promotion or completed bodies) is never re-verified and the corpus auditor does not compare H1 vs frontmatter title.
- **Why it is a workflow gap:** Post-gate title drift (the #432/#458 divergence shape) has no mechanical detection surface anywhere.
- **Confidence (emitter):** medium (formal candidate block; Claude alternatives-lens critic on the #1110 plan, seconded by the reconciler)
- **Triage evidence (2026-07-08):** NOT fixed on main: the gate-time H1-vs-frontmatter check landed in verify_task_body.py (commit 1dd959e650, issue-1110) but audit_clean_results_body_discipline.py has no H1-title-sync corpus check (grep for find_h1_title / 'frontmatter title' / H1: no relevant hits) — post-gate retitles on awaiting_promotion/completed bodies remain mechanically unwatched, exactly the parked ask. Completed #1015 targeted this file for a different bug (interval_inline) — not a dedup. No retraction.

## Proposed change (candidate diff sketch — refine in planning)

```
+ def audit_h1_title_sync(body, fm): ...  # WARN row per divergent sentinelled body
+   (reuse verify_task_body.find_h1_title + whitespace-collapse normalization)
+ wire into the auditor's per-body check list; WARN only, never FAIL
```

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Secondary: reuse (import or mirror) `scripts/verify_task_body.py`'s find_h1_title/normalization; add fixtures to the auditor's test file.
- Grep the workflow surface for the pattern before editing
  (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- origin: parked candidate on task #1110 at 2026-07-07T13:04:29Z

Verbatim parked note:

parked — running under workflow_fix_target Provenance (recursion guard); LOGGED for the next orchestrator/human pass, NOT auto-filed.

<!-- workflow-fix-candidate v1 -->
target_file: scripts/audit_clean_results_body_discipline.py
bug_observed: Retitles AFTER the final gate (on awaiting_promotion post-critic-PASS or completed bodies — the #432/#458 divergence shape) are never re-verified: no verifier run revisits those bodies, and the corpus auditor does not compare H1 vs frontmatter title.
why_workflow_gap: The #1110 check binds only at gate-time verify runs; post-gate title drift on grandfathered/parked bodies has no mechanical surface.
proposed_change: Add a corpus-wide WARN-level H1-vs-frontmatter-title comparison to audit_clean_results_body_discipline.py (reuse verify_task_body's find_h1_title + whitespace-collapse normalization; WARN only, never FAIL — post-gate remediation is a human call).
diff_sketch: |
  + def audit_h1_title_sync(body, fm): ...  # WARN row per divergent sentinelled body
  + (wire into the auditor's per-body check list)
confidence: medium
related_task: #1110
<!-- /workflow-fix-candidate -->

(Origin: Claude alternatives-lens critic concern on the #1110 plan, seconded by the reconciler as "a valid follow-up note".)

