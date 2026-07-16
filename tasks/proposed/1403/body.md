---
title: 'daily-fix: Step 10d conflicts go to fresh subagent'
kind: infra
tags:
- wf-fix
- wf-fix-fp:337f9d2baa93
- daily-auto-filed
created_at: '2026-07-16T07:21:43Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): #1338 hit real 4-file conflicts
  at Step 10d, paged the conflict diff inline into a near-full context, and died on
  ''Prompt is too long'' with no recovery turn'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

When a Step 10d squash-merge hits real content conflicts late in a session, dispatch the conflict investigation/resolution to a fresh worktree-scoped subagent instead of paging the conflict diff inline into a near-full orchestrator context.

## Workflow gap

- **Bug observed:** the #1338 session hit real 4-file content conflicts at Step 10d after a full pipeline run, paged the conflict diff inline into a near-full context, and died on "Prompt is too long" with no recovery turn (bc8a308f 16:12:54-16:14:51Z).
- **Why it is a workflow gap:** Step 10d's merge-conflict recovery ladder is written for the orchestrator to execute inline — there is no context-pressure branch that routes a late-session multi-file conflict to a fresh subagent, so the recovery itself can kill the session that must run it.
- **Severity:** medium
- verified-at-filing: `grep -c 'conflict.*subagent\|subagent.*conflict' .claude/skills/issue/SKILL.md` → 0 hits — no subagent-dispatch branch in the conflict recovery (absence confirmed); Step 10d heading at SKILL.md:8700, merge-conflict recovery machinery present around L8789-9127 (dirty-tree abort, Guard 1, re-snapshot retry) — all inline-orchestrator shaped (2026-07-16 UTC).

## Proposed change (refine in planning)

Extend `.claude/skills/issue/SKILL.md` Step 10d (L8700; merge-conflict recovery around L8789-9127): when the squash/rebase merge surfaces real content conflicts AND the session is late/context-heavy (heuristic: past the review rounds, or any conflict spanning >1-2 files), the orchestrator spawns a fresh `implementer`-class subagent scoped to the worktree with a lean brief (branch, conflicted paths from `git diff --name-only --diff-filter=U`, the resolution contract) instead of paging conflict bodies inline; the subagent resolves + commits, the orchestrator verifies and completes the merge. Diff-size-budget discipline (`.claude/rules/diff-size-budget.md`) applies to the brief.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 10d, L8700 + conflict recovery L8789-9127)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 337f9d2baa93

- workflow_fix_target: .claude/skills/issue/SKILL.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: bc8a308f (#1338) 16:12:54-16:14:51Z (batch 09 P1/P2).
