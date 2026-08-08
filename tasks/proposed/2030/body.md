---
title: 'workflow-fix: step9c gate stale-__pycache__ purge'
kind: infra
tags:
- wf-fix
- wf-fix-fp:82cc7365c5c1
- daily-auto-filed
created_at: '2026-08-03T06:40:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): The stale-`__pycache__`
  determinism class fixed for the inline gate in #1950 (same-second Edit → ruff-hook
  rewrite leaves a validation-passing stale pyc that plain `uv run pytest` children
  import) plausibly also affects the Step 9c test-verdict gate''s worktree pytest
  runs, which have no bytecode purge or `PYTHONDONTWRITEBYTECODE=1` threading.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-08-02 Step C parked-candidate routing pass from a
workflow-fix candidate parked on task #1950 (emitting agent: the #1950 round-1
implementer, via the orchestrator's recursion-guard park;
`epm:workflow-fix-candidate` at 2026-08-03T02:42:36Z).

## Goal

Extend the #1950 stale-`__pycache__` determinism fix (landed for
`scripts/inline_lint_gate.py`, merged PR #1728 squash 59a57db10e75) to the Step
9c test-verdict gate's worktree pytest runs: purge stale bytecode +
`PYTHONDONTWRITEBYTECODE=1` in the child env before the gate's pytest
invocation(s).

## Workflow gap

- **Bug observed:** the stale-`__pycache__` determinism class fixed for the inline gate in #1950 (same-second Edit → ruff-hook rewrite leaves a validation-passing stale pyc that plain `uv run pytest` children import) plausibly also affects the Step 9c test-verdict gate's worktree pytest runs, which have no bytecode purge or `PYTHONDONTWRITEBYTECODE=1` threading. `unverified hypothesis — verify at plan time:` that the Step 9c gate is actually susceptible in practice (the #1950 candidate marks itself confidence: low; the #1345 incident trace covered the inline gate only).
- **Why it is a workflow gap:** the workflow's other pytest-running gate (Step 9c) shares the exact import mechanics the #1345 incident traced, but the #1950 fix was deliberately scoped to `scripts/inline_lint_gate.py` only.
- **Confidence (emitter):** low
- verified-at-filing: `grep -n -E 'purge|DONTWRITEBYTECODE|__pycache__' scripts/step9c_baseline.py` → 0 hits; same pattern in `.claude/skills/issue/SKILL.md` → 0 hits; `grep -n 'def purge_repo_bytecode' scripts/inline_lint_gate.py` → 1 hit (line 262, the reusable helper); `git log --oneline --since='7 days ago' -- scripts/step9c_baseline.py` → 3 commits, none touching bytecode handling (2026-08-02). Absence claim about a guard, 0-hit in-target = the evidence (not a text-matching-guard subclass — the probe is for the mechanism identifiers themselves plus the landed-fix history check).

## Proposed change (candidate diff sketch — refine in planning)

```
+ from inline_lint_gate import purge_repo_bytecode  # or a shared helper
+ purge_repo_bytecode(worktree_root)  # before the Step 9c pytest dispatch
+ env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
```

Call-hop note (clause g): the Step 9c pytest DISPATCH is prescribed by
`.claude/skills/issue/SKILL.md` (the orchestrator runs the gate per its recipe)
and `scripts/step9c_baseline.py` runs the pristine-scratch oracle pytest — both
construction sites are in the target list; the planner picks where the purge
lands (a shared helper is plausible).

## Scope / surfaces

- Primary target: `scripts/step9c_baseline.py, .claude/skills/issue/SKILL.md`
- Grep the workflow surface for other un-purged pytest invocation sites before
  editing; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/step9c_baseline.py, .claude/skills/issue/SKILL.md
- fingerprint: 82cc7365c5c1

Origin candidate (verbatim, parked on #1950 at 2026-08-03T02:42:36Z):

<!-- workflow-fix-candidate v1 -->
target_file: scripts/step9c_baseline.py, .claude/skills/issue/SKILL.md
bug_observed: The stale-`__pycache__` determinism class fixed for the inline gate in #1950 (same-second Edit → ruff-hook rewrite leaves a validation-passing stale pyc that plain `uv run pytest` children import) plausibly also affects the Step 9c test-verdict gate's worktree pytest runs, which have no bytecode purge or `PYTHONDONTWRITEBYTECODE=1` threading.
why_workflow_gap: The workflow's other pytest-running gate (Step 9c) shares the exact import mechanics the #1345 incident traced, but the #1950 fix was deliberately scoped to `scripts/inline_lint_gate.py` only.
proposed_change: Reuse `purge_repo_bytecode` (or an equivalent purge + `PYTHONDONTWRITEBYTECODE=1` child-env guard) before the Step 9c gate's pytest invocation in the worktree.
confidence: low
related_task: #1950
<!-- /workflow-fix-candidate -->
