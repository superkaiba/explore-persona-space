---
title: 'daily-fix: choom-protect the Step 10d lint-gate first run'
kind: infra
tags:
- wf-fix
- wf-fix-fp:56b828b43014
- daily-auto-filed
created_at: '2026-07-09T07:01:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): #1143''s Step 10d pre-push
  lint gate leg died on first run (consistent with an OOM-score kill on the shared
  VM, #811 family) and passed on the choom-protected re-run — ~10-25 min lost.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Protect the Step 10d lint-gate legs from shared-VM OOM-score kills on the first run.

## Workflow gap

- **Bug observed:** Transcript fa0fb96a (issue-1143) 12:16:32Z: 'Lint gate verdict: crash' — the no-flags workflow-lint leg died mid-run; the re-run carried sudo -n choom -n -600 -p $$ and passed.
- **Why it is a workflow gap:** The just-landed #1138 canonical snippet does not carry choom protection, so every first gate run is exposed to the #811 kill family on the shared VM.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

Prefix the canonical lint-gate snippet with the choom self-protection line (same recipe as the #811 vectorized-fits rule), keeping sudo -n fail-soft.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-B P2 (fa0fb96a 12:16:32Z)
