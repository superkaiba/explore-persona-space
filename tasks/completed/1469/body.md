---
title: 'daily-fix: 6a.5 gate asserts git-tree reachability'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6fa92d54950b
- daily-auto-filed
created_at: '2026-07-17T06:58:48Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the Step 6a.5 carry-over-artifact
  gate reported ''all carry-over artifacts resolve'' while checking LOCAL resolution
  only — the #1434 cell manifest was uncommitted/untracked, so the git-clone GCP lane
  never staged it and all 12 runs failed TWICE with FileNotFoundError (#734 family;
  ~2 boot cycles + 24 run-attempts lost, fixed by committing the manifest f9f1002797)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1434 session 027e29e8, ~04:50-05:39Z).

## Goal

Make the pre-dispatch artifact gate catch uncommitted inputs before a git-clone-only lane boots.

## Workflow gap

- **Bug observed:** the Step 6a.5 carry-over-artifact gate reported 'all carry-over artifacts resolve' while checking LOCAL resolution only — the #1434 cell manifest was uncommitted/untracked, so the git-clone GCP lane never staged it and all 12 runs failed TWICE with FileNotFoundError (#734 family; ~2 boot cycles + 24 run-attempts lost, fixed by committing the manifest f9f1002797)
- **Why it is a workflow gap:** The #734 clause already names git-tree reachability as the reuse requirement; the gate that is supposed to enforce it checks the wrong predicate (local presence).
- **Confidence (emitter):** high (24 failed run-attempts today; fix confirmed by commit f9f1002797)
- verified-at-filing: incident in #1434 transcript (12 runs x2 FileNotFoundError; gate had reported all-resolve); commit f9f1002797 verified resolving via git log; `grep -n '6a.5' .claude/skills/issue/SKILL.md` locates the gate block (git-tree predicate absent — planning verifies the exact block)

## Proposed change (candidate diff sketch — refine in planning)

make the 6a.5 gate additionally assert each plan-referenced input under eval_results/ or data/ is git-tree-reachable at the dispatch tip when the lane is git-clone-only (per the existing #734 clause in artifact-reuse)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 6fa92d54950b

