---
title: 'daily-fix: planner reuse search live sibling worktrees'
kind: infra
tags:
- wf-fix
- wf-fix-fp:69e864d55605
- daily-auto-filed
created_at: '2026-07-16T07:20:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): Two live sessions (#1335/#1345)
  re-implemented modules #825 had already built + reviewed on its unmerged branch;
  Thomas had to flag the consolidation'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Extend the planner's reuse search from merged code + sibling issue branches to LIVE sibling sessions' in-flight worktree modules, surfacing an advisory before re-implementing something a concurrent session has already built.

## Workflow gap

- **Bug observed:** two live sessions (#1335/#1345 siblings) were re-implementing modules #825 had already built + reviewed on its unmerged branch; Thomas had to say "flag the consolidation" (09f28ede 22:39Z). The in-session claim that a "reuse-gap workflow-fix is filed" was NOT verified — the miners' grep of tasks/proposed+planning found no matching task (only #1072, an unrelated experiment).
- **Why it is a workflow gap:** planner.md's reuse search covers merged code, HF artifacts, and PAST sibling issue branches, but has no step that enumerates CONCURRENT live sessions' in-flight worktree modules — so parallel sessions on the same parent duplicate each other's builds.
- **Severity:** high
- verified-at-filing: `grep -n 'sibling' .claude/agents/planner.md` → 5 hits (L84, L104, L130-131, L149, L289 — all parent/sibling ISSUE reuse: bodies, plans, clean-results); `grep -n 'live session\|in-flight' .claude/agents/planner.md` → 0 hits — proposed live-sibling-worktree leg absent (2026-07-16 UTC); dedup re-check: `grep -rl 'reuse.*live\|in-flight.*module' tasks/proposed tasks/planning` → no open matching task (consistent with the miners' 07-15 grep).

## Proposed change (refine in planning)

Extend `.claude/agents/planner.md` steps 1/5 (reuse search): in addition to merged code + sibling issue branches, enumerate LIVE sibling sessions' in-flight worktree modules — worktree branches of sibling issues on the same parent (`git -C .claude/worktrees/issue-<M>* log --oneline origin/main..HEAD -- <module glob>`, plus `spawn_session.py list` for live issue mappings) — and surface an advisory in the plan before re-implementing an overlapping module. The advisory names the sibling issue + module and proposes reuse-or-consolidation instead of a fresh build (extends the "Reuse existing in-repo tools" hierarchy and the #722 built-but-stranded lesson to CONCURRENT work).

## Scope / surfaces

- Primary target: `.claude/agents/planner.md` (steps 1/5 reuse search; anchors L84, L130-131, L149)
- Secondary: possibly `.claude/rules/artifact-reuse.md` (the reuse checklist this feeds)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 69e864d55605

- workflow_fix_target: .claude/agents/planner.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 09f28ede 22:39Z (batch 08 P16); grep verification 2026-07-15 tonight: no open task matches.
