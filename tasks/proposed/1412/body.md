---
title: 'daily-fix: binary figure conflict guidance 9b/10d'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f96c8f84f9dc
- daily-auto-filed
created_at: '2026-07-16T07:22:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): PR #1066 merge failed on
  a binary-PDF conflict when a same-issue branch and a concurrent inline round both
  wrote figures/issue_1090/fu4/; each session improvises ~10 min of recovery'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Add Step 9b/10d guidance for binary figure conflicts: round-scoped figure regeneration prefers `checkout --theirs`-style resolution for binary figure conflicts (regenerable artifacts — take the newer round's copy).

## Workflow gap

- **Bug observed:** PR #1066's merge failed on a binary-PDF conflict when a same-issue branch and a concurrent inline round both wrote `figures/issue_1090/fu4/`; each session currently improvises the binary-conflict recovery (~10 min each; 9d362ba4 04:58:29Z).
- **Why it is a workflow gap:** Step 10d documents text/`tasks/` conflict recovery ladders and even NAMES binary `figures/` collisions as an expected fall-through case, but gives no resolution recipe for them — so every session re-derives "figures are regenerable, take the newer copy".
- **Severity:** low
- verified-at-filing: `grep -n 'binary' .claude/skills/issue/SKILL.md` → L9905 names "binary `figures/` collisions — #697/#597" as an expected skip-predicate fall-through in the conflict recovery, with NO resolution recipe following (the surrounding recovery text at L9898-9915 covers re-snapshot retries for `tasks/` conflicts only) — guidance absent, insertion point identified (2026-07-16 UTC).

## Proposed change (refine in planning)

Extend `.claude/skills/issue/SKILL.md` Step 9b/10d merge-conflict recovery (anchor: the binary `figures/` collisions mention at L9905): for a binary conflict under `figures/issue_<N>/` between a same-issue branch and a concurrent round, resolve with `git checkout --theirs -- <path>` (or equivalently take the NEWER round's regenerated copy) — figures are regenerable artifacts whose sidecar records provenance, so content-merge is meaningless and the newest regeneration wins; note the losing copy is recoverable from the other branch if ever needed.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 9b/10d conflict recovery; anchor L9905)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: f96c8f84f9dc

- workflow_fix_target: .claude/skills/issue/SKILL.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 9d362ba4 (#1090) 04:58:29Z (batch 07 P2).
