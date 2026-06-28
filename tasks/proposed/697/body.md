---
title: 'workflow-fix: align contradictory finding-less analysis routing prose in SKILL.md'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f4aad9b20d75
created_at: '2026-06-28T06:39:50Z'
has_clean_result: false
parent_id: 694
origin_prompt: 'Codex codex-code-reviewer r2 Style/Consistency note on task #694:
  SKILL.md:6034-6038 vs :6045-6047 say different things about whether a finding-less
  analysis task self-skips inside the agent or is routed straight to Step 10d. Non-blocking,
  but prose is slightly contradictory.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a Codex r2 code-reviewer follow-up surfaced on task #694 (emitting agent: codex-code-reviewer).

## Goal

Align the two prose locations in `.claude/skills/issue/SKILL.md` that currently disagree about how a finding-less `kind: analysis` task is routed at Step 10 — one says the task self-skips inside `related-work-finder`, the other says it's routed straight to Step 10d.

## Workflow gap

- **Bug observed:** `.claude/skills/issue/SKILL.md` carries contradictory prose at two routing-related locations (post-#694 line numbers: approximately the "Per-kind membership" / Step 10c-bis-gating block, vs the Step 10 step 10 entry-condition block). Codex r2: `SKILL.md:6034-6038` says a finding-less analysis task "that reaches this branch" self-skips inside the agent, while `SKILL.md:6045-6047` routes finding-less analysis straight to Step 10d.
- **Why it is a workflow gap:** the effective routing matches the plan (Step 10c-bis only fires for finding-bearing analysis), but the two prose locations being slightly contradictory invites a future reader to misread the contract. A surfaced concrete documentation inconsistency in a workflow-surface file is exactly the prose-follow-up class the workflow-fix-on-bug protocol auto-files.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

Pick ONE phrasing — likely "finding-less analysis never enters Step 10b/10c/10c-bis (Step 10's entry condition is `kind:experiment OR (kind:analysis AND has ## Results)`)" — and use it in BOTH locations. The "self-skip inside the agent" prose is now redundant because the entry condition already prevents the agent from being spawned for finding-less analysis. Delete the inner self-skip prose OR keep it as defense-in-depth but make the contract explicit: the entry condition is primary, the in-agent skip is the redundant backstop.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface before editing (`grep -rn 'finding-less' .claude/ CLAUDE.md`) and reconcile every hit to one consistent phrasing; list them in the plan.

## Constraints / invariants

- Workflow-surface only — `.claude/skills/issue/SKILL.md` is the only target.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes (no `.py` expected).
- The effective routing behavior must not change — this is prose alignment only.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: f4aad9b20d75

(Surfaced as Codex `codex-code-reviewer` r2 Style/Consistency note on task #694 — verbatim: "`.claude/skills/issue/SKILL.md:6034-6038` says a finding-less analysis task 'that reaches this branch' self-skips inside the agent, while `.claude/skills/issue/SKILL.md:6045-6047` routes finding-less analysis straight to Step 10d. Non-blocking, because the effective routing matches the round-2 requirement, but the prose is slightly contradictory.")
