---
title: revert/am fence sibling in repo-root branch guard
kind: infra
tags:
- wf-fix
- wf-fix-fp:f46ae29ce62c
- daily-auto-filed
created_at: '2026-07-10T06:54:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): The runtime detector''s
  verb list (scripts/guard_repo_root_branch.sh:566) covers checkout/switch/restore/clean/reset/merge/rebase/cherry-pick
  but not the revert or am verbs — two remaining ungated tree'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1193.

## Goal
Add a revert/am fence sibling mirroring the #1193 detectors, with an anchored allow for the abort/quit recovery subcommands.

## Workflow gap
- **Bug observed:** The runtime detector's verb list (scripts/guard_repo_root_branch.sh:566) covers checkout/switch/restore/clean/reset/merge/rebase/cherry-pick but not the revert or am verbs — two remaining ungated tree/history-mutating verbs on the same shared-root conflict-stranding threat model the #1128/#1193 fences close.
- **Why it is a workflow gap:** The guard hook is the runtime enforcement of the shared-repo-root protection; an ungated mutating verb family leaves the same clobber channel open.
- **Confidence (emitter):** low

## Proposed change (candidate diff sketch — refine in planning)
(none — emitter sketch: mirror the #1193 rebase/cherry-pick fence arms for the two remaining verbs, allow --abort/--quit recovery forms)

## Scope / surfaces
- Primary target: `scripts/guard_repo_root_branch.sh`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/guard_repo_root_branch.sh
- fingerprint: n/a (prose park)

source: prose-followup (planner §8, task #1193). target_file: scripts/guard_repo_root_branch.sh. bug_observed: git revert (and git am) remain ungated tree/history-mutating verbs on the same shared-root conflict-stranding threat model the #1128/#1193 fences close. proposed_change: a revert/am fence sibling mirroring the #1193 detectors (anchored allow for --abort/--quit recovery), if incident demand appears. confidence: low (zero incident demand today; named in register item (xvii)(a)). routed: parked — this session is a workflow-fix session (workflow_fix_target Provenance line on #1193; recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard) and never auto-files more workflow-fix tasks; parked for the nightly /daily parked-candidate routing pass. Distinct fingerprint from #1193 (different verb family).
