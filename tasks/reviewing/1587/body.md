---
title: 'daily-fix: dispatch-time trigger-dense tag auto-adoption'
kind: infra
tags:
- wf-fix
- wf-fix-fp:369acba0a151
- daily-auto-filed
created_at: '2026-07-22T06:44:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): guard-surface tasks rely
  on per-turn re-recognition of trigger-dense targets; nothing adopts a durable trigger-dense
  tag at dispatch time'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1574 (emitting agent: planner, plan v1 §2 item 4 post-plan note; ancestry #1556/#1546).

## Goal

Add dispatch-time auto-adoption of a durable trigger-dense task tag at `/issue` SKILL.md Step 6d.2, keyed on the `.claude/rules/trigger-dense-review.md` recognition heuristic, with a durability pin and Step-9c selector registration.

## Workflow gap

- **Bug observed:** guard-surface tasks rely on per-turn RE-recognition of trigger-dense targets (the #1563 Step-0 orchestrator-turn discipline at SKILL.md:375-388); nothing ADOPTS a durable trigger-dense tag at dispatch time, so the recognition is re-derived each turn and downstream consumers (Step-9c selector registration, review-round brief composition) cannot key on a durable signal.
- **Why it is a workflow gap:** the recognition heuristic is defined in `.claude/rules/trigger-dense-review.md` but SKILL.md Step 6d.2 has no tag-adoption step — a session crash/respawn or a subagent brief composer has no durable marker that the task is trigger-dense.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'add-tag.*trigger\|trigger-dense.*tag\|tag.*trigger-dense' .claude/skills/issue/SKILL.md` → 0 hits (2026-07-22, absence claim — the in-target 0-hit IS the evidence; the per-turn #1563 discipline exists at SKILL.md:375-388 but adopts no tag); `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` shows the #1563/#1546 discipline commits but no tag-adoption landing.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up) Dispatch-time auto-adoption of the trigger-dense tag at SKILL.md Step 6d.2: apply the trigger-dense-review.md recognition heuristic once at dispatch, `task.py add-tag <N> trigger-dense` (or equivalent durable signal), pin it for durability across respawns, and register the tag with the Step-9c selector so trigger-dense rounds are selector-visible.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'trigger-dense' .claude/ CLAUDE.md scripts/`) and update every hit that needs to key on the new tag; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 369acba0a151

- workflow_fix_target: .claude/skills/issue/SKILL.md

Verbatim parked candidate (task #1574 events, 2026-07-21T06:58:22Z): "parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see .claude/rules/workflow-fix-on-bug.md § Recursion guard; NOT auto-routed. source: prose-followup (planner plan v1 §2 item 4 post-plan note). target_file: .claude/skills/issue/SKILL.md. proposed_change: dispatch-time auto-adoption of the trigger-dense tag at SKILL.md Step 6d.2 (recognition heuristic per .claude/rules/trigger-dense-review.md + durability pin + Step-9c selector registration). confidence: medium. related_task: #1574 (deferred from plan v1; ancestry #1556/#1546)."
