---
title: 'daily-fix: 9c selector rules-to-prose-pin-test mapping'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1b7505ec017d
- daily-auto-filed
created_at: '2026-07-18T06:46:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): select_step9c_tests.py
  does not map .claude/rules/*.md edits to their prose-pin tests — rules edits short-circuit
  into the generic WORKFLOW_INVARIANT set, so per-rule pin coverage is coincidental.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-17 parked-candidate sweep (Step C) from a prose-followup candidate parked on task #1468 (emitting agent: code-reviewer round-1 verdict body; parked under the recursion guard).

## Goal

Make `scripts/select_step9c_tests.py` map `.claude/rules/*.md` edits to their specific prose-pin tests, instead of relying on the rules glob short-circuiting into the generic WORKFLOW_INVARIANT set.

## Workflow gap

- **Bug observed:** the Step 9c test selector does not map `.claude/rules/*.md` edits to their prose-pin tests — in #1468's round both such tests happened to run and pass only because they are members of the generic WORKFLOW_INVARIANT set, not because the selector targeted them from the touched rule file.
- **Why it is a workflow gap:** a rule file whose prose-pin test is NOT in the WORKFLOW_INVARIANT set gets no targeted coverage on a rules-only diff; the mapping is coincidental, so a future prose-pin test added outside the invariant list silently stops gating the rule it pins.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "rules/" scripts/select_step9c_tests.py` → 1 hit at line 212, inside `WORKFLOW_SURFACE_GLOBS` (the short-circuit set: "gate via the WORKFLOW_INVARIANT set, not a per-file test") — confirming rules edits short-circuit rather than map to per-rule pin tests (2026-07-18 UTC). No per-rule pin mapping exists in the file (no other `rules/` hit).

## Proposed change (candidate diff sketch — refine in planning)

Add a rules→pin-test mapping pass (e.g. by test-name convention `tests/test_*<rule-stem>*` or an explicit map) that ENUMERATES the matching prose-pin tests for touched `.claude/rules/*.md` files in addition to the invariant set; a rules edit whose pin test exists but is unmapped should surface it.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Related sibling being filed the same night: the literal-path pinning-test-shape gap (same file, distinct bug/fingerprint — see the daily-fix filing from #1483's implementer-r2 candidate). The spawned sessions should cross-reference to avoid overlapping edits.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes; selector tests green.
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 1b7505ec017d

- workflow_fix_target: scripts/select_step9c_tests.py

source candidate (verbatim, prose park on #1468, 2026-07-17T17:49:06Z): "Suggestion: scripts/select_step9c_tests.py does not map .claude/rules/*.md edits to their prose-pin tests (both such tests happened to run + pass this round)."
