---
title: 'workflow-fix: lint: pin awk elision program identical across'
kind: infra
tags:
- wf-fix
- wf-fix-fp:59c93cc64393
- daily-auto-filed
created_at: '2026-07-09T06:57:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The ban-gate awk elision
  program is duplicated verbatim in two workflow-surface files with no mechanical
  check that the copies stay byte-identical, so an edit to one home silently drifts
  the other.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #998 (park_form: recursion-guard).

## Goal

Add a ~5-line workflow_lint check pinning the ban-gate awk elision program byte-identical across its two full-text homes (.claude/skills/issue/SKILL.md 9a-humanize and .claude/rules/analyzer-section-reference.md Step 4.5).

## Workflow gap

- **Bug observed:** The ban-gate awk elision program is duplicated verbatim in two workflow-surface files with no mechanical check that the copies stay byte-identical, so an edit to one home silently drifts the other.
- **Why it is a workflow gap:** The elision program is executable text agents copy-paste at run time; divergent copies mean the humanize ban-gate behaves differently depending on which file the agent read.
- **Confidence (emitter):** medium (code-reviewer round-1 Minor, #998)

## Proposed change (candidate diff sketch — refine in planning)

Extract the awk program block from both files (between stable fence markers), assert byte equality, FAIL naming both paths; bundle into the no-flags default run.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- origin: parked candidate on task #998 at 2026-07-04T22:45:46Z

parked — running under workflow_fix_target Provenance (recursion guard, workflow-fix-on-bug.md § Recursion guard). source: prose-followup (code-reviewer round-1 Minor). target_file: scripts/workflow_lint.py. proposed_change: add a ~5-line lint check pinning the ban-gate awk elision program byte-identical across the two full-text homes (.claude/skills/issue/SKILL.md 9a-humanize + .claude/rules/analyzer-section-reference.md Step 4.5). confidence: medium. Not auto-routed; next human/orchestrator pass may file it.
