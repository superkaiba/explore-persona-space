---
title: 'daily-fix: piped-git guard ignores quoted pattern text'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0250cc57eb91
- daily-auto-filed
created_at: '2026-07-25T06:49:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): guard_piped_git_push.sh
  false-blocked a read-only grep whose quoted pattern literal contained ''git commit
  -m'' piped to head - no git command in the pipeline - during the 07-23 /daily run,
  costing a wasted turn and a pattern rewrite to evade the hook'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session d2e3d231, the 07-23 /daily run).

## Goal

The piped-git guard should block real piped git/gh commands, not quoted strings that merely mention them.

## Workflow gap

- **Bug observed:** the 07-23 /daily session ran `grep -n "bare.*commit\|git commit -m" scripts/workflow_lint.py | head -5` (read-only) and the hook blocked it ("BLOCKED: piping `git push` / `git merge` / `git commit`...", 1 firing 06:32:20Z); the session rephrased the grep pattern to evade the hook and moved on.
- **Why it is a workflow gap:** the guard matches the verb substring anywhere in the command line, including inside single/double-quoted argument text, so any grep/echo that mentions `git commit` in a pattern trips it — a false-positive class that trains sessions to evade the hook.
- **Confidence (emitter):** high on the FP; medium on the exact anchoring implementation (shell parsing is subtle; planner decides regex vs word-boundary+position heuristic).
- verified-at-filing: `.claude/hooks/guard_piped_git_push.sh` header confirms substring intent (#1048/#1591 lineage); `git log --oneline --since='7 days ago' -- .claude/hooks/guard_piped_git_push.sh` → no FP fix landed (only 75873f3b7b #1591 widening) (2026-07-25).

## Proposed change (candidate diff sketch — refine in planning)

Anchor the guarded-verb match to pipeline-segment command position (segment start after `&&`, `;`, `|`, `(`), or strip quoted spans before matching — keeping every TRUE-positive shape from #1048/#1584/#1591 blocked (extend the hook's test cases with the quoted-pattern FP + the true positives).

## Scope / surfaces

- Primary target: `.claude/hooks/guard_piped_git_push.sh`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 0250cc57eb91

- workflow_fix_target: .claude/hooks/guard_piped_git_push.sh
