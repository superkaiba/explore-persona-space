---
title: 'workflow-fix: verify_plan cross-section param-consistency check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:62da7cf04f7c
created_at: '2026-07-05T06:06:11Z'
has_clean_result: false
origin_prompt: 'codex-critic methodology twin on #1024: recurring verifier for stale
  post-fact-check restatements (temperature=0.7 vs corrected temperature-omitted across
  plan sections)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1024 (emitting agents: codex-critic methodology twin, ratified by the Claude methodology critic, statistics critic, and consistency-checker — all four independently re-flagged the same stale restatement).

## Goal

Add a cross-section internal-consistency check to `scripts/verify_plan.py` flagging contradictory `key=value` parameter restatements across plan sections (the stale-post-correction-restatement class).

## Workflow gap

- **Bug observed:** plan v2 for #1024 carried `temperature=0.7` in a §11 Decision-Rationale *What:* line contradicting the fact-check-corrected §4 D-E text ("temperature OMITTED / API default 1.0"); the mechanical pre-pass (0 FAIL/0 WARN) could not see it, and three critics + the consistency-checker each burned tokens independently re-flagging it.
- **Why it is a workflow gap:** a fact-check/critique correction applied to one plan section routinely leaves stale restatements of the same parameter in other sections (§11 What-lines, §10 repro commands); an implementer grounding from the wrong section threads the wrong value. Nothing mechanical catches cross-section contradictions today.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ In scripts/verify_plan.py, add check c21_cross_section_param_consistency:
+   - extract `\b(temperature|max_tokens|lr|epochs|seed|rank|alpha|batch(_size)?)\s*[=:]\s*<value>` tokens per section
+   - for any parameter appearing with >1 DISTINCT value across sections, WARN naming both lines,
+     unless one occurrence sits in a "declared but never threaded"/"historical"/"was/old" clause
+   - WARN not FAIL (legit multi-value cases exist: sweeps, per-phase values)

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'verify_plan' .claude/ CLAUDE.md scripts/`) and update every doc naming the check list; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; tests for the new check added to `tests/test_verify_plan.py`.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 62da7cf04f7c

Surfaced prose (codex-critic methodology twin, #1024 round 1): "This should recur in a workflow-surface verifier for stale post-fact-check restatements. ... check the plan text after fact-check corrections for any diagnosis/probe instruction matching temperature=0.7 or temperature: 0.7 outside a 'declared but never threaded' historical clause, and fail if found."
