---
title: 'daily-fix: codex-code-reviewer Step 0.9 + tag parity'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cf92b41f34ed
- daily-auto-filed
created_at: '2026-07-17T06:51:00Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): two parity gaps vs code-reviewer.md:
  the Step 0.9 git-provenance self-check recipe exists only as the compressed Blocker-tags
  parenthetical (L692), absent from the copy-list and the enumeration; and the data-access-blocked
  tag prescribed at L679 is not enumerated in the L692 Blocker-tags template line'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from two parked prose candidates on task #1380 (alternatives-critic Phase 2 + code-reviewer r1). Note #1380's own merge (ef84e6b19a) added the Step 4.6 Gate-scope parity, NOT these two.

## Goal

Reconcile the codex-code-reviewer composed-prompt rubric with code-reviewer.md on the Step 0.9 recipe and the data-access-blocked Blocker-tags enumeration.

## Workflow gap

- **Bug observed:** two parity gaps vs code-reviewer.md: the Step 0.9 git-provenance self-check recipe exists only as the compressed Blocker-tags parenthetical (L692), absent from the copy-list and the enumeration; and the data-access-blocked tag prescribed at L679 is not enumerated in the L692 Blocker-tags template line
- **Why it is a workflow gap:** The Codex twin's composed rubric is the only spec the Codex reviewer sees; an un-copied step or an un-enumerated tag silently degrades its verdicts and the reconciler's tag parsing.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n '0\.9' .claude/agents/codex-code-reviewer.md` -> 1 hit (L692, the Blocker-tags parenthetical only — no copy-list bullet / enumeration entry, the absence claim); `grep -n 'data-access-blocked' .claude/agents/codex-code-reviewer.md` -> hits at L668/L675/L679 (prescription) with the L692 Blocker-tags enumeration listing marker-shape/smoke-run-missing/git-provenance/raw-completions-upload-missing/cached-artifact-coverage-unverified/compute-shape-mismatch/hollow-verification-gate/substantive and NOT data-access-blocked; `git log --oneline --since='7 days ago' -- .claude/agents/codex-code-reviewer.md` -> ef84e6b19a (#1380 Step 4.6 parity — different gap)

## Proposed change (candidate diff sketch — refine in planning)

Two edits: (1) add a Step 0.9 copy-list bullet + '0.9' to the enumeration (or a 'deliberately template-only' note beside the L692 tag); (2) add `data-access-blocked` to the L692 Blocker-tags template enumeration.

## Scope / surfaces

- Primary target: `.claude/agents/codex-code-reviewer.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: cf92b41f34ed



