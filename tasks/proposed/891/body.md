---
title: 'workflow-fix: thread caps at VM launch surfaces despite stale worktrees'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6783ad72aa1b
created_at: '2026-07-02T21:00:43Z'
has_clean_result: false
origin_prompt: 'PM-chat root-cause 2026-07-02: pre-#847 worktree launched 78 uncapped
  threads; no propagation or launch-time fallback for merged infra fixes.'
---
## Overview / Motivation
Auto-filed from the 2026-07-02 #779 uncapped-grid launch: the #847 thread-cap fix (env.py OMP setdefault for shared-VM launches) merged to main, but the issue-779 worktree branched earlier and kept the pre-fix env.py; the session's VM-side grid launch ran 78 uncapped threads. The worktree HAS a spec-freshness sync mechanism for .claude/ files (commit e81357fd "sync workflow-surface specs from main") but nothing propagates src/ library infra fixes, and launch instructions do not require explicit caps.

## Goal
Close the worktree-staleness hole for shared-infra fixes: VM-side compute launches from a worktree must either (a) explicitly export the thread-cap envs in the launch command (belt-and-suspenders, works regardless of branch age), or (b) the spec-freshness sync is extended to a small allowlist of infra-critical src files (orchestrate/env.py), applied before any VM-side CPU-phase dispatch.

## Workflow gap
- **Bug observed:** a fix that existed on main for ~20h was absent from the running worktree; the launch surface had no fallback, so the exact failure #847 fixed recurred.
- **Why it is a workflow gap:** in-flight worktrees are pinned to their branch point by design; infra fixes with fleet-wide blast radius (thread caps) have no propagation or launch-time fallback path.
- **Confidence (emitter):** medium

## Proposed change (refine in planning)
- `.claude/skills/issue/SKILL.md` (VM-side phase dispatch) + `.claude/agents/experiment-implementer.md`: VM-side CPU-phase launch commands MUST carry explicit `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8` (values from CLAUDE.md/code-style), independent of worktree branch age.
- Optionally extend the spec-freshness sync allowlist with `src/explore_persona_space/orchestrate/env.py` (planner decides whether the sync or the explicit-caps route is the cleaner invariant).

## Scope / surfaces
- Primary: `.claude/skills/issue/SKILL.md`, `.claude/agents/experiment-implementer.md`, possibly the spec-freshness sync helper.

## Constraints / invariants
- Caps are explicit-launch-time only (pods/GCE/SLURM keep full width); no library behavior change without the planner's say.

## Provenance
- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: PM-chat root-cause 2026-07-02 (#779 uncapped grid from a pre-#847 worktree)
