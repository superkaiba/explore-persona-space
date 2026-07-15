---
title: 'workflow-fix: c6_reuse_fitness recognizes (a)-(j) reuse-map tables'
kind: infra
tags:
- wf-fix
- wf-fix-fp:03e9bf7c9986
created_at: '2026-07-15T06:10:09Z'
has_clean_result: false
origin_prompt: 'Methodology critic prose follow-up on #1090 fu5 plan v7: mechanizable
  c6 table-form false positive'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1090 (emitting agent: critic [Methodology lens], fu5 plan-critique round).

## Goal

Teach `verify_plan.py` check 6 (`c6_reuse_fitness`) to recognize an (a)-(j)-keyed reuse-map TABLE as a fitness-check declaration.

## Workflow gap

- **Bug observed:** c6_reuse_fitness WARNed on plan v7 of #1090 whose §4 D3 table carries a complete (a)-(j)-keyed artifact-reuse fitness map — format false positive ("plan reuses HF artifacts but no fitness check found").
- **Why it is a workflow gap:** the check's lettered-item detector (`check_reuse_fitness`, scripts/verify_plan.py:831-869) spots lettered items in some shapes but missed a markdown table whose Checks column enumerates `(a)`/`(e)`/`(f)`/`(h)(i)-(iii)`/`(j)` per row; every future plan using the table form (the natural shape for multi-artifact reuse) re-trips the WARN and burns a critic-adjudication pass.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "c6_reuse_fitness" scripts/verify_plan.py` → 3 hits in 1 file (function at :831, FAIL text at :869, registry at :5698); the pass-branch at :855 counts "lettered items spotted", confirming the detector exists and the gap is its table-form coverage (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

In `check_reuse_fitness` (scripts/verify_plan.py:831), extend the lettered-item detection to also match checklist letters inside markdown table rows, e.g.:
+ table_letters = set(re.findall(r"\(([a-j])\)", "\n".join(l for l in plan.splitlines() if l.lstrip().startswith("|"))))
+ letters |= table_letters
(exact regex up to the planner; the intent is: an (a)-(j)-keyed reuse-map table row counts toward the letters threshold at :855)

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'c6_reuse_fitness' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; `tests/test_verify_plan.py` extended with a table-form fixture.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 03e9bf7c9986

Surfaced prose (verbatim, Methodology critic on #1090 fu5 plan v7): "mechanizable: yes — c6 should recognize a `(a)–(j)`-keyed reuse-map table as a fitness check (regex for checklist letters in a §4/§10/§11 table row), not only a fixed heading."
