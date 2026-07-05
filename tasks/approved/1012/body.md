---
title: 'workflow-fix: reconcile stale ensemble round ranges (1-3) with the cap-5 policy'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e10511d8abc4
created_at: '2026-07-04T15:54:10Z'
has_clean_result: false
origin_prompt: 'codex-clean-result-critic composer #928 r4 prose follow-up: spec hard-codes
  rounds 1-3 while the ensemble round cap is 5; a valid round-4 delta re-gate fell
  outside the stated range — reconcile live prescriptions across the workflow surface,
  leave historical notes verbatim'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #928 (emitting agent: codex-clean-result-critic composer, round-4 delta re-gate, 2026-07-04).

## Goal

Update the live ensemble-round ranges that hard-code "every round (1-3)" to reflect the per-reviewer round cap of 5 (all rounds), distinguishing live prescriptions from historical policy notes.

## Workflow gap

- **Bug observed:** `.claude/agents/codex-clean-result-critic.md` hard-codes "spawned in parallel ... on EVERY round (1-3)" while the project ensemble policy caps iterating review sites at 5 rounds per reviewer; a legitimate round-4 delta re-gate on task #928 fell outside the spec's stated range, and the composer had to deviate from its own spec on explicit orchestrator instruction (it composed rather than refused, and flagged the stale range).
- **Why it is a workflow gap:** the agent spec and the ensemble policy (round cap 5, all-rounds ensembling since 2026-06-12) disagree; a stricter composer could refuse a valid round-4/5 dispatch, stalling the final gate.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
- ... on **EVERY round (1-3)** (all-rounds policy as of 2026-06-12; previously round-1-only).
+ ... on **EVERY round (1 through the round cap, 5)** (all-rounds policy as of 2026-06-12; previously round-1-only).
```

Apply the analogous edit wherever the "(1-3)" range is a LIVE prescription (spawn instructions, ensemble policy statements); leave historical notes ("round-1-only until 2026-06-12") untouched.

## Scope / surfaces

- Primary target: `.claude/agents/codex-clean-result-critic.md`
- Grep hits to triage (live prescription vs historical note; the frozen deprecated `codex-reviewer.md` is excluded): `.claude/agents/codex-clean-result-critic.md`, `.claude/agents/clean-result-critic.md`, `.claude/skills/issue/SKILL.md`, `.claude/skills/clean-results/iterations.md`, `.claude/rules/agents-vs-skills.md`, `.claude/workflow.yaml`, `CLAUDE.md` (pattern: `round (1-3)|rounds (1-3)|rounds 1-3|round-1-only`). Same triage applies to the sibling `interpretation-critic` / `codex-interpretation-critic` descriptions if they carry the same live range.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` default run passes; if `workflow.yaml` changes, regenerate any auto-generated tables (`workflow_lint.py --emit-tables`).
- Do NOT change the round cap itself (5, per-reviewer, reconciler excluded) — only the stale "(1-3)" ranges that contradict it.
- Historical policy notes stay verbatim.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/codex-clean-result-critic.md
- fingerprint: e10511d8abc4

Verbatim surfaced prose (codex-clean-result-critic composer, #928 r4): "One note for the record: my agent spec still hard-codes rounds 1-3 while the project-level ensemble policy caps rounds at 5 — I followed the orchestrator's explicit round-4 brief (composing, not refusing) since the spec's range predates the cap-5 policy; flagging the stale range in `.claude/agents/codex-clean-result-critic.md` as a concrete workflow-surface follow-up for you to route."
