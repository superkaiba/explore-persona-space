---
title: 'workflow-fix: lint: marker-recipe doc numeric snippets match'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d6da23c77908
- daily-auto-filed
created_at: '2026-07-09T06:57:13Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Frozen numeric snippets
  in docs/marker_training_recipe.md index rows can silently drift from the cited code
  constants; no mechanical consistency check exists in workflow_lint.py.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1001 (park_form: recursion-guard).

## Goal

Add a workflow_lint check that the frozen numeric snippets in docs/marker_training_recipe.md index rows stay consistent with the code constants they cite.

## Workflow gap

- **Bug observed:** Frozen numeric snippets in docs/marker_training_recipe.md index rows can silently drift from the cited code constants; no mechanical consistency check exists in workflow_lint.py.
- **Why it is a workflow gap:** A stale frozen number in the recipe index misleads every future marker-training planner grounding hyperparameters from the doc.
- **Confidence (emitter):** low (Codex statistics S6, optional)

## Proposed change (candidate diff sketch — refine in planning)

Add a check_marker_recipe_snippets pass: parse the index rows' numeric snippets in docs/marker_training_recipe.md, resolve each cited constant in the named source file, FAIL on mismatch.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- origin: parked candidate on task #1001 at 2026-07-05T00:26:43Z

parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). Two prose follow-ups raised by the Phase 2 critics, logged NOT routed:
1. (both alternatives critics) Durable sibling fix: wire organisms.enforce_mix_token_budget as a DEFAULT at the shared train_lora / marker-mix-assembly seam so non-reader implementers are mechanically protected (today the gate is wired only in issue906_phase1_pilot.py's _assemble_marker_mix). target_file: src/explore_persona_space/train/sft.py (+ shared mix builders). Separate kind:infra task for a future orchestrator pass.
2. (Codex statistics S6, optional/low) A mechanizable workflow-surface check that frozen numeric snippets in docs/marker_training_recipe.md index rows stay consistent with cited code constants. target_file: scripts/workflow_lint.py.
