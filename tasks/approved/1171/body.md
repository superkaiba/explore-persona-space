---
title: 'workflow-fix: strip v4 Methodology details in body audit'
kind: infra
tags:
- wf-fix
- wf-fix-fp:21a56a6abeaa
- daily-auto-filed
created_at: '2026-07-09T06:58:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): strip_data_example_blocks
  (audit_clean_results_body_discipline.py:505) drops <details> example blocks ONLY
  inside the v3 `## Data` section (re-enters only on `## Data`, :539-540) — v4 `##
  Methodology` <details> sample-data blocks are never stripped, so verbatim example
  rows carrying condition codes / interval forms can false-positive audit categories
  on v4 bodies. [merged sibling: strip_data_examp'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #969 (recursion-guarded workflow-fix session).

## Goal

Stop v4 clean-result bodies from false-positiving the body-discipline audit on verbatim sample-data rows that live in ## Methodology by spec.

## Workflow gap

- **Bug observed:** strip_data_example_blocks (audit_clean_results_body_discipline.py:505) drops <details> example blocks ONLY inside the v3 `## Data` section (re-enters only on `## Data`, :539-540) — v4 `## Methodology` <details> sample-data blocks are never stripped, so verbatim example rows carrying condition codes / interval forms can false-positive audit categories on v4 bodies.
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/audit_clean_results_body_discipline.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
- # re-enter only when the new H2 is `## Data` itself
+ # re-enter when the new H2 is `## Data` (v3) or `## Methodology` (v4 sample-data slot)
- in_section = line.strip() == "## Data"
+ in_section = line.strip() in ("## Data", "## Methodology")
```

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- origin: parked candidate on task #969 at 2026-07-04T07:45:46Z

Verbatim parked note:

> routed: parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard); NOT auto-filed. source: prose-followup (planner assumption-12 observation, #969 Phase 1). target_file: scripts/audit_clean_results_body_discipline.py. bug_observed: strip_data_example_blocks gates on '## Data' only (line ~496), so v4 '## Methodology' <details> sample-data blocks are NOT stripped for ANY audit category — verbatim example rows carrying condition codes / interval forms can false-positive on v4 bodies. proposed_change: extend strip_data_example_blocks (or add a v4 branch) to also strip <details> blocks under v4 ## Methodology. confidence: medium. related_task: #969.


### Merged sibling candidate (s4-audit-v4-sample-block-exempt, from task:970 at 2026-07-04T07:53:33Z)

- bug_observed: strip_data_example_blocks exempts <details> example blocks only inside the v3 `## Data` section; v4 bodies carry spec-mandated verbatim sample blocks under `## Methodology` -> `**Sample training/evaluation data + completions:**`, which therefore remain in the scan source for EVERY non-exempt category — verbatim rows cannot be reworded, so a caps token in an unfenced v4 sample block can false-FAIL the audit.
- proposed_change: Extend the <details> exemption to the v4 Methodology sample slot for ALL scan categories (mirror the existing ## Data mechanism), with a v4-details-stays-clean test plus a #763-Results-still-flags test.
- origin note (verbatim): NOT fixed: scripts/audit_clean_results_body_discipline.py `strip_data_example_blocks` (l.505-552) still enters <details> drop-mode ONLY inside the v3 `## Data` section ('Any H2 ends a ## Data block... re-enter only when the new H2 is ## Data itself'); no `## Methodology` / v4 sample-slot handling anywhere. The completed audit-script fixes since (#1015 interval_inline bound-form; #969; #731; #892) are unrelated. Distinct (target_file, fingerprint 770f63a8859c) from #970's own fix. No retraction in #970 events.
