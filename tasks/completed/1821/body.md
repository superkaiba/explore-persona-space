---
title: 'daily-fix: bake self-pid exclusion into single-flight probe'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b31a9cd2b342
- daily-auto-filed
created_at: '2026-07-29T07:19:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1742 re-hit the documented
  #1606 single-flight self-match trap (the probe folded inside the launch call matched
  itself) despite the SKILL.md rule mandating a separate FOREGROUND call — remembered
  rules keep being re-hit'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-I P6 (rule anchor miner-probed; re-verified).

## Goal

Replace the remembered single-flight probe-placement rule with a mechanical self-pid-excluding probe.

## Workflow gap

- **Bug observed:** #1742's session folded the Step 9c single-flight probe inside the launch call; the probe matched its own wrapper (the documented #1606 trap) despite SKILL.md line 7329's explicit 'in a separate FOREGROUND call — never inside' rule. One more re-hit of a rule that exists precisely because it keeps being forgotten.
- **Why it is a workflow gap:** the defense is prose placement discipline; a self-pid-excluding probe (helper or entrypoint-internal) removes the failure mode regardless of placement.
- **Confidence (emitter):** medium-high
- verified-at-filing: `grep -n 'separate FOREGROUND call' .claude/skills/issue/SKILL.md` → line 7329 (rule present, trap still re-hit) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

A helper (or step9c_baseline.py entry-internal probe) with `-P $$`/pgid exclusion + bracket pattern; SKILL.md snippet updated to call it.

## Scope / surfaces

- Primary targets: `scripts/step9c_baseline.py`, `.claude/skills/issue/SKILL.md` (Step 9c 1b snippet)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: b31a9cd2b342

- workflow_fix_target: scripts/step9c_baseline.py

