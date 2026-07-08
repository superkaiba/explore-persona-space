---
title: 'workflow-fix: codex twin round-bound 1-3 vs cap-5 delta rounds'
kind: infra
tags:
- wf-fix
- wf-fix-fp:88eb7e26dc46
created_at: '2026-07-04T20:00:22Z'
has_clean_result: false
origin_prompt: 'codex-clean-result-critic composer #952 r4 prose follow-up'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by codex-clean-result-critic (composer) on task #952 r4.

## Goal

Relax the `revision_round` bound in `.claude/agents/codex-clean-result-critic.md` + `.claude/agents/codex-interpretation-critic.md` from 1-3 to 1-5, with rounds 4-5 valid only as delta-scoped re-reviews after a reconciler-bound REVISE; refuse only malformed rounds (≤0, >5, non-integer).

## Workflow gap

- **Bug observed:** The #952 r4 delta dispatch — a legitimate post-reconciler round under the cap-5 ensemble policy (workflow.yaml § ensemble_review round cap 5) — violated the twin specs' stated 1-3 round bound ("epm:failure on anything else"); the composer had to proceed on a self-declared stale-spec assumption.
- **Why it is a workflow gap:** the twin specs' round bound predates the all-rounds/cap-5 policy; any future round-4/5 delta review re-hits the contradiction, and a literal-minded composer would falsely post epm:failure on a valid dispatch.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
- rounds 1-3 only; post epm:failure on anything else
+ rounds 1-5; rounds 4-5 are valid only as delta-scoped re-reviews following a
+ reconciler-bound REVISE (workflow.yaml round cap 5); refuse only malformed
+ rounds (<=0, >5, non-integer)
```

## Scope / surfaces

- Primary target: `.claude/agents/codex-clean-result-critic.md, .claude/agents/codex-interpretation-critic.md`
- Grep first: `grep -rln 'revision_round' .claude/agents/codex-*.md` and align every twin with the same stale bound.

## Constraints / invariants

- Workflow-surface only; keep the delta-scoped framing consistent with workflow.yaml § ensemble_review.

## Provenance

- workflow_fix_target: .claude/agents/codex-clean-result-critic.md, .claude/agents/codex-interpretation-critic.md
- fingerprint: 88eb7e26dc46
