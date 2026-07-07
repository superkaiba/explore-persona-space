---
title: 'workflow-fix: check-17 verbatim equality for origin_prompt footer quote'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9dc0f6bb1a35
created_at: '2026-07-05T12:49:27Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #813 r1 prose follow-up: extend verify_task_body.py
  check 17 to enforce normalized verbatim equality between the Context footer originating-prompt
  blockquote and frontmatter origin_prompt (prefix-truncation FAIL)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised by the clean-result-critic on task #813 (emitting agent: clean-result-critic, round 1 of the per-example-vs-averaged-map follow-up gate).

## Goal

Extend `verify_task_body.py` check 17 to enforce normalized verbatim equality between the `**Context:**` footer's "Originating prompt, verbatim" blockquote and the frontmatter `origin_prompt`, FAILing on prefix-truncation.

## Workflow gap

- **Bug observed:** task #813's re-folded clean-result footer carried an "Originating prompt, verbatim" blockquote that was a strict mid-sentence prefix of the frontmatter `origin_prompt` (cut after "SAVE+UPLOAD the pre and post trained maps,", dropping the unreduced-activations directive) — and check 17 PASSed it; only the LM-side Lens 5 review caught it.
- **Why it is a workflow gap:** the spec's check 17 verifies the Context row ships the originating prompt, but does not compare the quoted text against `origin_prompt`, so "verbatim" is unenforced mechanically — any truncation survives to the final gate and burns an ensemble REVISE round.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

In `scripts/verify_task_body.py` check 17: extract the blockquote under the "Originating prompt, verbatim" label, normalize whitespace on both sides (collapse internal runs, strip), and require equality with normalized frontmatter `origin_prompt`; FAIL with a `context-origin-prompt-mismatch` message naming the first divergence offset. Keep a NO-OP pass when frontmatter `origin_prompt` is absent (legacy tasks). Add regression tests in tests/test_verify_task_body.py (equal → PASS; prefix-truncated → FAIL; absent frontmatter → NO-OP).

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Also: `tests/test_verify_task_body.py`, and the SPEC.md check-17 prose if its wording needs the equality rule named.
- Grep the workflow surface for check-17 references before editing (`grep -rln 'check 17\|origin_prompt' scripts/verify_task_body.py .claude/skills/clean-results/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; SPEC.md stays consistent with the verifier.
- Forward-compatible: v3/v2/legacy bodies are never newly hard-FAILed (gate the new equality FAIL on the v4 sentinel, WARN-only below it).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 9dc0f6bb1a35

Surfaced prose (clean-result-critic, task #813 r1): "the fix is a one-line append (mechanizable — a check-17 normalized-equality extension is sketched in the verdict note for the orchestrator to route)". Full verdict: /tmp/issue-813-clean-result-critique-r1.md (epm:clean-result-critique v1 on #813, 2026-07-05).
