---
title: 'workflow-fix: reviewer must verify seam-stubbed production bodies vs real
  APIs'
kind: infra
tags:
- wf-fix
created_at: '2026-07-03T18:53:45Z'
has_clean_result: false
origin_prompt: 'Auto-filed from #906 cap-5 incident: 5 review rounds shipped crash-class
  wrong-API production code behind seam stubs; reviewer spec lacks a production-body
  signature-verification rule.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from task #906 (emitting agent: orchestrator).

## Goal

Add a mandatory code-reviewer check: any NEWLY-ADDED production function whose tests stub/monkeypatch it (seam pattern) must have its BODY's external API calls verified against the real signatures (inspect.signature / read the callee) before PASS.

## Workflow gap

- **Bug observed:** On #906, five consecutive ensemble review rounds shipped crash-class production code (nonexistent dataclass field dereference, fabricated judge-call signature, missing required kwargs) because the test suite's PilotSeams stub/monkeypatch design validates DISPATCH, not production BODIES — and the Claude reviewer PASSed all 5 rounds (reconciler-overturned 4 times) by crediting resolver tests as body coverage.
- **Why it is a workflow gap:** `.claude/agents/code-reviewer.md` has no rule directing the reviewer at seam-hidden production bodies; the hollow-verification-gate check (Step 0.68) covers gates, not the body-vs-dispatch distinction.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ In code-reviewer.md (review rubric, near Step 0.68):
+ **Seam-stubbed production bodies:** for every production function ADDED in
+ this round that any test stubs/monkeypatches (grep test file for its name),
+ read the BODY and verify each external call against the callee's REAL
+ signature (`uv run python -c "import inspect; ..."`) and each attribute
+ dereference against the real dataclass fields. A dispatch/resolver test is
+ NOT body coverage. Wrong-signature / nonexistent-field findings are
+ Critical (substantive).

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md`
- Grep the workflow surface for sibling review specs that inherit the rubric (`grep -rln 'hollow-verification' .claude/agents/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under the standard (non-workflow-fix) context; the spawned session carries the recursion guard.

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md
- fingerprint: (computed by wrapper)

Surfaced prose: #906 rounds 1-5, reconciler v1-v5 markers (Claude PASS overturned r1/r2/r4/r5; root pattern named in epm:review-reconcile v5: "tests validate that the dispatcher calls those names, not that the bodies are correct").
