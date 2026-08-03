---
title: 'daily-fix: lint bare git-commit recipes lacking pathspec'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1318d69480dd
- daily-auto-filed
- trigger-dense
created_at: '2026-07-24T06:47:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): no mechanical check flags
  fenced bare git commit -m lines lacking a trailing pathspec in workflow-surface
  recipes, the 7dbde267f1 staged-sweep class'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-23 parked-candidate routing pass (Step C). Raised as a recursion-guard-parked candidate on task #1630 (alternatives-lens critic, plan v1 review).

## Goal

Add a `workflow_lint.py` check flagging fenced bare `git commit -m` lines that lack a trailing ` -- <pathspec>` in workflow-surface recipe files, so the shared-root staged-sweep hazard class (incident commit `7dbde267f1`, 2026-07-21) is guarded mechanically instead of per-file.

## Workflow gap

- **Bug observed:** other workflow-surface recipes may still prescribe bare `git commit -m` at the repo root; only /daily was fixed by #1630 and there is no mechanical check for the class.
- **Why it is a workflow gap:** CLAUDE.md § Concurrent repo-root committers tightens only the STAGING side; no lint flags fenced bare-commit lines lacking a pathspec, so the class stays open per-file (a bare commit on the always-concurrent shared root sweeps sibling sessions' staged files — the `7dbde267f1` incident swept 4 foreign files onto main).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "pathspec\|bare_commit\|check-daily-commit" scripts/workflow_lint.py` → no existing check of this class (absence claim; the two hits at lines 9337/9460 are the repo-root-branch-guard checkout forms, unrelated to commit pathspec discipline) (2026-07-24 UTC). Incident SHA `7dbde267f1` rev-parse-verified at compose time.

## Proposed change (candidate diff sketch — refine in planning)

New workflow_lint check (bundled into the no-flags default run): scan workflow-surface `.md` fenced code blocks for `git commit -m` lines with no trailing ` -- ` pathspec; allowlist the deliberate exceptions (scratch-worktree commits, `git -C` non-root forms, `--dry-run`).

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py` (+ `tests/test_workflow_lint.py` pin)
- Sweep the surface for current offenders before enabling; fix or allowlist each hit in the same change.

## Constraints / invariants

- Workflow-surface only. The new check must pass clean on the current tree when it lands (fix/allowlist all pre-existing hits).
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 1318d69480dd

- workflow_fix_target: scripts/workflow_lint.py

Origin: parked candidate on #1630 (2026-07-23T15:07:43Z).
