---
title: 'daily-fix: skip Codex composer spawns under live quota senti'
kind: infra
tags:
- wf-fix
- wf-fix-fp:81ffc6bcd5d1
- daily-auto-filed
created_at: '2026-07-09T07:01:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): With the #1126 quota sentinel
  active (until 2026-08-06), ~9+ sessions today still spawned 3-5 codex-* composer
  wrapper subagents per review round whose composed prompts were discarded at the
  dispatch short-circuit.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Stop spawning Codex prompt-composer subagents when the org-quota sentinel is already known-exhausted.

## Workflow gap

- **Bug observed:** The sentinel is checked at DISPATCH time (codex_task.py), but the thin codex-* composers are spawned in the same batch as the Claude critics BEFORE any dispatch — 3 composer spawns per plan round + 1 per code-review round run to completion and are discarded. Observed in >=9 sessions on 2026-07-08 (e.g. transcripts fa0fb96a ~11:0xZ, 5ce345bc 15:08-15:30Z, e96ed309 08:52Z); #1135 and #1146 skipped correctly, showing the intended behavior.
- **Why it is a workflow gap:** Wasted subagent spawns fleet-wide for a month-long outage; the skill text never tells the orchestrator to pre-check the sentinel before composing.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

In the Phase-2 / Step-5 / 9a / 9a-bis spawn-batch prose: `if [ -f .claude/cache/codex-quota-exhausted-until ] and the parsed reset is in the future: skip all codex-* composer spawns this round; post the no-show fallback marker directly.` Exit-9 short-circuit stays as backstop.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, .claude/skills/adversarial-planner/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, .claude/skills/adversarial-planner/SKILL.md
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-A P4, mine-B P3, mine-C P1, mine-D P2 (4 independent miners)
